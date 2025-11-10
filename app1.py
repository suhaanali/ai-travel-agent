import os
import re
import json
import random
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, jsonify, Response, session, current_app
from flask_cors import CORS
from amadeus import Client
import uuid

# ==========================================================
#  CONFIGURATION & INITIALIZATION
# ==========================================================

# --- Always load .env from the same directory as this file ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# --- Read the key from .env ---
flask_key = os.environ.get("FLASK_SECRET_KEY")

# --- Create Flask app BEFORE any helpers use 'session' ---
app = Flask(__name__)

# --- Set secret key ---
if flask_key:
    app.secret_key = flask_key
    print(f"[INFO] ✅ FLASK_SECRET_KEY loaded from .env ({len(flask_key)} chars)")
else:
    print("[WARN] ⚠️ No FLASK_SECRET_KEY found — generating temporary key.")
    app.secret_key = os.urandom(24)

# --- Enable CORS with credentials for session cookies ---
CORS(app, supports_credentials=True)

# --- Confirm key actually attached ---
print("[DEBUG] Flask secret key active:", bool(app.secret_key))

# --- Conversation Memory ---
conversation_history = []
MAX_HISTORY = 10


# =========================================================
# 🧭 UNIVERSAL DATE NORMALIZER — Fixes "January 2024" issue
# =========================================================
def normalize_future_date(dep_date_raw):
    """
    Ensures that all parsed flight dates are in the realistic future.
    Handles cases like:
      • User says "January 10th" in November 2025 → 2026-01-10
      • User says "December 5th" after it has passed → next year
      • Keeps explicit future years (e.g. 2026) intact
    """
    from datetime import datetime, date

    today = date.today()
    d = None

    # --- Try multiple formats safely ---
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            d = datetime.strptime(str(dep_date_raw).split("T")[0], fmt).date()
            break
        except Exception:
            continue

    if not d:
        # fallback if parsing fails
        return today

    # --- Smart Year Logic ---
    if d.year < today.year:
        # e.g. GPT gave 2024 when now is 2025 → move to next year
        d = d.replace(year=today.year + 1)
    elif d.year == today.year and d < today:
        # same year but date has already passed → bump to next year
        d = d.replace(year=today.year + 1)
    # if GPT gave a future year (e.g. 2026) → leave unchanged

    return d

# ==========================================================
#  SESSION CONTEXT HELPERS  (place right after initialization)
# ==========================================================
def get_context():
    """Retrieve the current user's stored session context."""
    return session.get("user_context", {})


def update_context(new_data):
    """Update and persist user context in Flask session."""
    ctx = session.get("user_context", {})
    ctx.update(new_data)
    session["user_context"] = ctx
    return ctx



# Utility for converting words to numbers
WORD_TO_NUMBER = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
}

# Common cities → IATA mapping
CITY_TO_IATA = {
    "mumbai": "BOM",
    "london": "LHR",
    "paris": "CDG",
    "new york": "JFK",
    "dubai": "DXB",
    "abu dhabi": "AUH",
    "shanghai": "PVG",
    "delhi": "DEL",
    "bangalore": "BLR"
}

# --- AMADEUS SDK CLIENT INITIALIZATION (Sandbox) ---
try:
    AMADEUS = Client(
        client_id=os.getenv("AMADEUS_API_KEY"),
        client_secret=os.getenv("AMADEUS_API_SECRET"),
        hostname="test"   # 👈 ensures sandbox (test.api.amadeus.com)
    )
    AMADEUS_TOKEN_STATUS = True
    print("[INIT] ✅ Amadeus client initialized successfully (Sandbox)")
except Exception as e:
    print(f"[ERROR] ❌ Amadeus Client Initialization Failed: {e}")
    AMADEUS = None
    AMADEUS_TOKEN_STATUS = False

import requests
import os

def get_weather(city):
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    if not API_KEY:
        print("[ERROR] OPENWEATHER_API_KEY is missing!")
        return "⚠️ Weather API key not configured."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    print(f"[DEBUG] Fetching weather for {city} → {url}")

    try:
        response = requests.get(url, timeout=10)
        print(f"[DEBUG] Status code: {response.status_code}")
        print(f"[DEBUG] Response text: {response.text[:300]}")

        if response.status_code == 200:
            data = response.json()
            desc = data["weather"][0]["description"].capitalize()
            temp = data["main"]["temp"]
            feels = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            return f"{desc}, {temp}°C (feels like {feels}°C, humidity {humidity}%)"
        elif response.status_code == 404:
            return f"⚠️ City '{city}' not found. Please check the spelling."
        else:
            return f"⚠️ Weather service returned error {response.status_code}."
    except Exception as e:
        print("[ERROR] Weather fetch failed:", e)
        return "⚠️ Could not fetch weather right now."


from datetime import datetime
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import pytz

def get_time(city_name: str):
    """Return the current local time for any city using geopy + timezonefinder."""
    try:
        geolocator = Nominatim(user_agent="smart_travel_ai")
        location = geolocator.geocode(city_name)

        if not location:
            return f"⚠️ Sorry, I couldn't find '{city_name}'. Please check the spelling."

        # Get timezone from lat/lon
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=location.longitude, lat=location.latitude)

        if not timezone_str:
            return f"⚠️ Sorry, I couldn’t find the timezone for {city_name}."

        tz = pytz.timezone(timezone_str)
        local_time = datetime.now(tz)
        formatted_time = local_time.strftime("%I:%M %p")

        return f"The local time in {city_name.title()} is {formatted_time} 🕒"

    except Exception as e:
        print(f"[ERROR] get_time() failed: {e}")
        return f"⚠️ Sorry, I couldn’t get the time for {city_name} right now."








def get_iata_code(city_name):
    """
    Resolves a city or airport name to a 3-letter IATA code.

    Priority:
    1️⃣ Direct recognition if user enters a valid IATA code (e.g. DXB)
    2️⃣ Fast local lookup for common global cities
    3️⃣ GPT-powered fallback for rare cities or typos
    """
    import re

    if not city_name:
        return None

    city_name = city_name.strip().upper()

    # ✅ 1. Direct IATA code recognition
    if re.fullmatch(r"[A-Z]{3}", city_name):
        print(f"[IATA] Provided valid IATA code: {city_name}")
        return city_name

    # ✅ 2. Local quick map (fastest and avoids unnecessary API/GPT calls)
    known_iata = {
        "DUBAI": "DXB",
        "ABU DHABI": "AUH",
        "SHARJAH": "SHJ",
        "DOHA": "DOH",
        "MUSCAT": "MCT",
        "RIYADH": "RUH",
        "JEDDAH": "JED",
        "MUMBAI": "BOM",
        "DELHI": "DEL",
        "BANGALORE": "BLR",
        "CHENNAI": "MAA",
        "KOCHI": "COK",
        "KARACHI": "KHI",
        "ISLAMABAD": "ISB",
        "LONDON": "LHR",
        "PARIS": "CDG",
        "FRANKFURT": "FRA",
        "NEW YORK": "JFK",
        "SINGAPORE": "SIN",
        "TOKYO": "HND",
        "ZURICH": "ZRH",
        "COPENHAGEN": "CPH",
        "BANGKOK": "BKK",
        "MANILA": "MNL",
        "TORONTO": "YYZ"
    }

    if city_name in known_iata:
        code = known_iata[city_name]
        print(f"[LOOKUP] {city_name} → {code}")
        return code

    # ✅ 3. GPT fallback for rare or misspelled cities
    try:
        prompt = f"Return only the 3-letter IATA airport code for the city '{city_name}'. Example: Paris → CDG."
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.2,
        )
        raw = response.output[0].content[0].text.strip().upper()
        if re.fullmatch(r"[A-Z]{3}", raw):
            print(f"[GPT IATA] {city_name} → {raw}")
            return raw
        else:
            print(f"[GPT WARN] Invalid IATA result for '{city_name}': {raw}")
            return None
    except Exception as e:
        print(f"[GPT ERROR] Failed to resolve IATA for {city_name}: {e}")
        return None





# --- Load environment variables ---
load_dotenv()
AMADEUS_KEY = os.getenv("AMAD_API_KEY")
AMADEUS_SECRET = os.getenv("AMAD_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- Initialize GPT client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Persistent memory store (simple dict for demo) ---
user_context = {}
conversation_state = {
    "stage": None  # can be: awaiting_origin, awaiting_destination, awaiting_date
}


# --- Default Amadeus settings ---
AMADEUS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"
DEFAULT_CURRENCY = "AED"

# --- Hotel placeholder ---
HOTEL_COMING_SOON_TEXT = (
    "🏨 The hotel booking feature is coming soon! "
    "You'll be able to search and book hotels right here."
)


# ===============================================================
# 🔹 Helper Functions
# ===============================================================

import requests
from datetime import datetime
import pytz

def get_weather(city):
    """Fetch live weather data using OpenWeatherMap API."""
    API_KEY = "YOUR_OPENWEATHER_API_KEY"  # get it from openweathermap.org
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        return f"{desc}, {temp}°C"
    return "unavailable right now."


from amadeus import ResponseError
from datetime import datetime, timedelta

def query_amadeus_flights(params):
    try:
        # --- Extract & normalize core params ---
        origin_raw = params["originLocationCode"]
        dest_raw = params["destinationLocationCode"]
        dep_date = params["departureDate"]
        adults = int(params.get("adults", 1))
        children = int(params.get("children", 0))
        travel_class = str(params.get("travelClass", "ECONOMY") or "ECONOMY").upper()
        currency = params.get("currencyCode", "AED")

        # ✈️ Always ensure we send valid IATA codes (even if city names were passed)
        origin = get_iata_code(origin_raw)
        dest = get_iata_code(dest_raw)

        if not origin or not dest:
            print(f"[IATA ERROR] Could not resolve one or both city names: origin={origin_raw}, destination={dest_raw}")
            return {"data": [], "error": "Invalid origin or destination"}

        print(f"[DEBUG] Querying Amadeus for {origin} → {dest} on {dep_date}")

        api_params = {
            "originLocationCode": origin,
            "destinationLocationCode": dest,
            "departureDate": dep_date,
            "adults": adults,
            "travelClass": travel_class,
            "currencyCode": currency,
            "max": 3
        }

        # ✅ Include optional flags only when needed
        if str(params.get("nonStop", "")).lower() == "true":
            api_params["nonStop"] = True
        if params.get("returnDate"):
            api_params["returnDate"] = params["returnDate"]

        # --- Call Amadeus API ---
        response = AMADEUS.shopping.flight_offers_search.get(**api_params)
        print(f"[DEBUG] Amadeus API Status: {response.status_code}")

        if response.status_code == 200:
            return {"data": response.data, "dictionaries": response.result.get("dictionaries", {})}
        else:
            print(f"[ERROR] Amadeus API returned {response.status_code}")
            print(f"[DETAILS] {response.result}")
            return {"data": [], "error": "No flights found or invalid parameters."}

    except ResponseError as e:
        print(f"[ERROR] Amadeus ResponseError: {e}")
        return {"data": [], "error": str(e)}




def get_airline_logo_url(carrier_code):
    """Return a consistent airline logo URL using carrier code."""
    return f"https://content.airhex.com/content/logos/airlines_{carrier_code}_100_100_s.png"

def gpt_fallback_response(user_query):
    """
    Uses OpenAI to generate a smart conversational answer for non-flight queries.
    """
    import json

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI travel companion. "
                    "Answer questions conversationally, accurately, and helpfully. "
                    "You can discuss travel, geography, culture, weather, distance, and general facts. "
                    "If asked about people, describe them neutrally. "
                    "If asked about locations, provide useful context or data. "
                    "Keep replies short (1-3 sentences)."
                ),
            },
            {"role": "user", "content": user_query},
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] GPT fallback failed: {e}")
        return "I'm sorry, I couldn't retrieve that information right now."


from datetime import datetime, timedelta
import re

# Keep track of last search context
user_context = {}

def adjust_relative_date(text, context):
    """
    Detect phrases like 'next day', 'previous day', '2 days later' etc.
    and adjust last departure/return date.
    """
    text = text.lower()

    # Determine which date to modify (departure or return)
    target_key = "departureDate"
    if "return" in text or "come back" in text:
        target_key = "returnDate"

    # Must have previous context
    if target_key not in context or not context[target_key]:
        return context  # nothing to modify

    # Parse last date
    last_date = datetime.strptime(context[target_key], "%Y-%m-%d")

    # Match patterns like "next day", "previous day", "2 days later"
    if "next day" in text or "day after" in text:
        new_date = last_date + timedelta(days=1)
    elif "previous day" in text or "day before" in text:
        new_date = last_date - timedelta(days=1)
    else:
        # Match "2 days later" or "3 days earlier"
        match = re.search(r"(\d+)\s+days?\s+(later|after|earlier|before)", text)
        if match:
            days = int(match.group(1))
            direction = match.group(2)
            new_date = last_date + timedelta(days=days if direction in ["later", "after"] else -days)
        else:
            return context  # no pattern found

    context[target_key] = new_date.strftime("%Y-%m-%d")
    return context


# === GLOBAL CONTEXT (to persist last trip info) ===
user_context = {}

def handle_user_query(user_input):
    global user_context
    parsed = parse_user_intent(user_input)
    text = user_input.lower()

    # 1️⃣ If this is a follow-up like "return on 15 nov"
    if "return" in text and re.search(r"\d{1,2}\s*[a-zA-Z]{3,}", text):
        # Extract date
        match = re.search(r"(\d{1,2}\s*[a-zA-Z]{3,})", text)
        if match:
            parsed_date = datetime.strptime(match.group(1) + f" {datetime.now().year}", "%d %b %Y")
            user_context["returnDate"] = parsed_date.strftime("%Y-%m-%d")

        # 👇 Auto-fill missing data from previous one-way search
        if user_context.get("origin") and user_context.get("destination"):
            # Reverse the trip automatically for return
            origin, destination = user_context["origin"], user_context["destination"]
            user_context["origin"], user_context["destination"] = destination, origin

        parsed.update(user_context)
        return build_flight_query(parsed)

    # 2️⃣ Handle relative time changes (“next day”, etc.)
    if re.search(r"next day|previous day|days? (later|after|before|earlier)", text):
        user_context = adjust_relative_date(user_input, user_context)
        parsed.update(user_context)
        return build_flight_query(parsed)

    # 3️⃣ Save/merge context for normal queries (“auh-bom 12 nov”)
    for key in ["origin", "destination", "departureDate"]:
        if parsed.get(key):
            user_context[key] = parsed[key]
    parsed.update({k: v for k, v in user_context.items() if k not in parsed})

    return build_flight_query(parsed)







def format_amadeus_response(amadeus_response, params):
    """
    ✅ FINAL VERSION — Currency-Aware + Fare Breakdown
    -------------------------------------------------
    • Works for one-way, round-trip, and multi-city
    • Always includes numeric price and currency
    • Adds detailed per-offer fare breakdown (adults/children/infants)
    • Auto-distributes prices when Amadeus omits travelerPricings
    """
    from datetime import datetime

    offers = amadeus_response.get("data", []) or []
    dictionaries = amadeus_response.get("dictionaries", {}) or {}
    formatted_flights = []

    # --- Helpers -------------------------------------------------------
    def get_carrier_name(code):
        return dictionaries.get("carriers", {}).get(code, code)

    def format_time(ts):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M")
        except Exception:
            return ts or ""

    def format_date(ts):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%d %b %Y")
        except Exception:
            return ts or ""

    # --- Main Loop -----------------------------------------------------
    for offer in offers:
        itineraries = offer.get("itineraries", [])
        if not itineraries:
            continue

        # ===============================================================
        # 💰 Normalize Price
        # ===============================================================
        raw_price = offer.get("price", {})
        price_total = 0.0
        price_currency = params.get("currencyCode", "AED")

        try:
            if isinstance(raw_price, dict):
                raw_total = raw_price.get("grandTotal") or raw_price.get("total") or "0"
                price_currency = raw_price.get("currency") or price_currency
            elif isinstance(raw_price, str):
                import re
                raw_total = re.sub(r"[^\d.]", "", raw_price)
            else:
                raw_total = "0"

            price_total = float(str(raw_total).replace(",", "").strip() or 0)
        except Exception as e:
            print(f"[WARN] Price normalization failed: {e}")
            price_total = 0.0

        price = {"total": price_total, "currency": price_currency}

        # ===============================================================
        # 💰 Build Fare Breakdown (Adults / Children / Infants)
        # ===============================================================
        pax_adults = int(params.get("adults", 1))
        pax_children = int(params.get("children", 0))
        pax_infants = int(params.get("infants", 0))

        traveler_pricings = offer.get("travelerPricings", []) or []
        adult_total = child_total = infant_total = 0.0

        if traveler_pricings:
            for tp in traveler_pricings:
                ttype = tp.get("travelerType", "").upper()
                tprice = float(tp.get("price", {}).get("total") or 0)
                if ttype == "ADULT":
                    adult_total += tprice
                elif ttype == "CHILD":
                    child_total += tprice
                elif ttype == "INFANT":
                    infant_total += tprice

        # ===============================================================
        # 🧠 Fallback — Split total evenly across all passengers
        # ===============================================================
        if not traveler_pricings or (child_total == 0 and infant_total == 0):
            total_pax = pax_adults + pax_children + pax_infants
            if total_pax > 0:
                share = price_total / total_pax
                adult_total = share * pax_adults
                child_total = share * pax_children
                infant_total = share * pax_infants
            else:
                adult_total = price_total

        fare_breakdown = {
            "currency": price_currency,
            "adults": {"count": pax_adults, "total": round(adult_total, 2)},
            "children": {"count": pax_children, "total": round(child_total, 2)},
            "infants": {"count": pax_infants, "total": round(infant_total, 2)},
            "grand_total": round(price_total, 2)
        }

        # ===============================================================
        # ✈️ Extract flight segment details
        # ===============================================================
        def extract_segment_info(itinerary):
            segs = itinerary.get("segments", [])
            if not segs:
                return None

            first_seg, last_seg = segs[0], segs[-1]
            via_points = [s["arrival"]["iataCode"] for s in segs[:-1]] if len(segs) > 1 else []
            stops = len(segs) - 1
            duration = (
                itinerary.get("duration", "")
                .replace("PT", "")
                .replace("H", "h ")
                .replace("M", "m")
            )

            try:
                bag_info = (
                    offer["travelerPricings"][0]["fareDetailsBySegment"][0]
                    .get("includedCheckedBags", {})
                )
                bag_weight = f"{bag_info.get('weight', 25)} {bag_info.get('weightUnit', 'KG')}"
            except Exception:
                bag_weight = "25 KG"

            carrier = first_seg.get("carrierCode", "")
            return {
                "origin": first_seg["departure"]["iataCode"],
                "destination": last_seg["arrival"]["iataCode"],
                "departureTime": format_time(first_seg["departure"]["at"]),
                "departureDate": format_date(first_seg["departure"]["at"]),
                "arrivalTime": format_time(last_seg["arrival"]["at"]),
                "arrivalDate": format_date(last_seg["arrival"]["at"]),
                "flightDuration": duration,
                "carrierName": get_carrier_name(carrier),
                "carrierCode": carrier,
                "logoUrl": get_airline_logo_url(carrier),
                "flightNumber": f"{carrier}{first_seg.get('number', '')}",
                "stops": stops,
                "via": ", ".join(via_points) if via_points else "Direct",
                "cabin": (
                    offer.get("travelerPricings", [{}])[0]
                    .get("fareDetailsBySegment", [{}])[0]
                    .get("cabin", "ECONOMY")
                ),
                "aircraft": first_seg.get("aircraft", {}).get("code", "N/A"),
                "baggage": bag_weight,
            }

        outbound_info = extract_segment_info(itineraries[0])
        inbound_info = extract_segment_info(itineraries[1]) if len(itineraries) > 1 else None

        flight_obj = {**(outbound_info or {}), "price": price, "fare_breakdown": fare_breakdown}
        if inbound_info:
            flight_obj["returnFlight"] = inbound_info

        formatted_flights.append(flight_obj)

        # 💬 Debug each fare
        print(f"[DEBUG FARE] {flight_obj.get('flightNumber')} → {fare_breakdown}")

    # --- Summary -------------------------------------------------------
    print(f"[DEBUG] ✅ Parsed {len(formatted_flights)} flights successfully.")
    return formatted_flights








# ===============================================================
# 🔹 GPT Interpreter
# ===============================================================

def interpret_query_with_gpt(query, context=None):
    """
    🌍 Ultra-Smart Multilingual Flight & Travel Interpreter (v5)
    ------------------------------------------------------------
    Handles:
      ✅ One-way, return, open-jaw, and multi-city trips
      ✅ Complex date ranges and flexibility ("between Dec 10–15")
      ✅ 'Via' routing, stopovers, alliance preferences
      ✅ Passenger counts, cabin classes, and child/infant handling
      ✅ Airline preferences (Emirates, Star Alliance, etc.)
      ✅ Multilingual queries with English defaults
      ✅ Context-aware follow-up (e.g. “from Abu Dhabi?”)
    """
    from datetime import datetime, timedelta
    import json, re

    system_prompt = """
You are an expert multilingual travel planning AI that converts user flight queries into structured JSON for flight search APIs.

Your task:
- Detect the user's true travel intent (flights, multi-city, hotels, weather, etc.)
- Understand natural phrases, dates, routes, and constraints.
- Output ONLY valid JSON (no explanations, no markdown).

----------------------------------------------------
🧭 RECOGNIZED INTENTS
----------------------------------------------------
- "search_flights"        → one-way or round-trip flight
- "search_multicity"      → multiple legs or 'via' cities
- "search_hotels"         → hotels, stays, resorts
- "search_weather"        → weather or forecast queries
- "search_time"           → timezones or time differences
- "search_general"        → people, brands, or factual info
- "general_chat"          → greetings or small talk

----------------------------------------------------
🧩 JSON OUTPUT FORMAT
----------------------------------------------------
{
  "intent": "<intent>",
  "segments": [
    {"origin": "<city>", "destination": "<city>", "departureDate": "<YYYY-MM-DD or null>"}
  ],
  "origin": "<city or null>",
  "destination": "<city or null>",
  "departureDate": "<YYYY-MM-DD or null>",
  "returnDate": "<YYYY-MM-DD or null>",
  "travelClass": "<ECONOMY|BUSINESS|FIRST|PREMIUM_ECONOMY or null>",
  "nonStop": "<true|false|null>",
  "adults": "<int>",
  "children": "<int>",
  "infants": "<int>",
  "preferredAirlines": ["<airline names>"],
  "alliances": ["<Star Alliance|Oneworld|SkyTeam>"],
  "maxStops": "<int or null>",
  "text": "<brief natural English summary of the user’s request>"
}

----------------------------------------------------
🧠 BEHAVIOR RULES
----------------------------------------------------
1. If user lists several cities (e.g. "Dubai → Singapore → Tokyo → Dubai") → intent = "search_multicity"
2. If user mentions "via" or "stopover" → treat as "search_multicity" with extra segments.
3. If user says "return from another city" (open-jaw), make 2 segments.
4. Detect and interpret date ranges like:
   - “between Dec 10–15” → use start = Dec 10, end = Dec 15
   - “returning Jan 5–10” → use Jan 10 for returnDate
5. If “flexible” or “+/- N days” → still output a fixed date, but mention flexibility in `text`.
6. Detect passenger counts and cabin types:
   - “2 adults, 1 child, 1 infant” → adults=2, children=1, infants=1
   - “business”, “first”, “premium economy”, etc.
7. Detect non-stop/direct preference → nonStop=true
8. Detect alliance or airline constraints:
   - “Star Alliance”, “Qatar Airways”, “Emirates”, “Air India”, etc.
9. Maintain English replies unless query is fully in another language.
10. Never output plain text — JSON only.
"""

    try:
        messages = [{"role": "system", "content": system_prompt}]

        # Context awareness
        if context:
            ctx = f"Previous context: {json.dumps(context, ensure_ascii=False)}"
            messages.append({"role": "system", "content": ctx})

        # Add user query
        messages.append({"role": "user", "content": query})

        # 🔮 Call GPT model
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(completion.choices[0].message.content)

        # =====================================================
        # 🧠 POST-PROCESSING FIXES
        # =====================================================
        today = datetime.now().date()

        # --- Normalize date ranges like "between Dec 10–15"
        def parse_date_range(text):
            text = text.lower()
            m = re.search(r'(\d{1,2})\D{0,3}[-toand]*\D{0,3}(\d{1,2})\s*(\w+)', text)
            if m:
                d1, d2, month = m.groups()
                try:
                    month_num = datetime.strptime(month[:3], "%b").month
                    year = today.year + (1 if month_num < today.month else 0)
                    start = datetime(year, month_num, int(d1)).date()
                    end = datetime(year, month_num, int(d2)).date()
                    return start.isoformat(), end.isoformat()
                except:
                    return None, None
            return None, None

        dep_start, dep_end = parse_date_range(query)
        ret_start, ret_end = parse_date_range(query.split("return")[-1]) if "return" in query.lower() else (None, None)
        if dep_start and not parsed.get("departureDate"):
            parsed["departureDate"] = dep_start
        if ret_end and not parsed.get("returnDate"):
            parsed["returnDate"] = ret_end

        # --- Handle "via" routing if segments missing
        if parsed.get("intent") == "search_multicity" and not parsed.get("segments"):
            via_match = re.findall(r"via\s+([\w\s,]+)", query.lower())
            if via_match:
                via_cities = [x.strip().title() for x in re.split(r",|or|and", via_match[0]) if x.strip()]
            else:
                via_cities = []
            origin, destination = parsed.get("origin"), parsed.get("destination")
            all_cities = [origin] + via_cities + [destination] if origin and destination else []
            parsed["segments"] = [
                {"origin": all_cities[i], "destination": all_cities[i + 1], "departureDate": parsed.get("departureDate")}
                for i in range(len(all_cities) - 1)
            ]

        # --- Fill return trips for open-jaw routes
        if parsed.get("intent") == "search_flights" and not parsed.get("segments"):
            if parsed.get("origin") and parsed.get("destination"):
                parsed["segments"] = [
                    {"origin": parsed["origin"], "destination": parsed["destination"], "departureDate": parsed.get("departureDate")},
                ]
                if parsed.get("returnDate"):
                    parsed["segments"].append({
                        "origin": parsed["destination"],
                        "destination": parsed["origin"],
                        "departureDate": parsed["returnDate"]
                    })

        # --- Normalize class
        if parsed.get("travelClass"):
            cls = parsed["travelClass"].upper().replace(" ", "_")
            if "PREMIUM" in cls and "ECONOMY" not in cls:
                cls = "PREMIUM_ECONOMY"
            parsed["travelClass"] = cls

        # --- Fill safe defaults
        for key, val in {
            "adults": 1, "children": 0, "infants": 0,
            "nonStop": None, "preferredAirlines": [], "alliances": [], "maxStops": None
        }.items():
            parsed.setdefault(key, val)

        print(f"[GPT PARSED] {json.dumps(parsed, indent=2)}")
        return parsed

    except Exception as e:
        print("[ERROR] GPT interpretation failed:", e)
        return {
            "intent": "general_chat",
            "segments": [],
            "origin": None,
            "destination": None,
            "departureDate": None,
            "returnDate": None,
            "travelClass": None,
            "nonStop": None,
            "text": "Hello! How can I assist you with your travel plans today?"
        }






import re
from datetime import datetime, timedelta
from dateutil import parser

def normalize_date(text, reference_date=None):
    """
    Converts user or GPT date inputs like '30 Nov', '04 Jan', 'today', '2023-11-30'
    into one or more ISO date strings (YYYY-MM-DD).

    ✅ Handles departure + return dates (e.g. 'on 19 Nov and return on 23 Dec')
    ✅ Uses current or next year intelligently.
    ✅ Prevents old years like 2023 from GPT.
    ✅ If date already passed this year → automatically bumps to next year.
    ✅ Handles cross-year trips correctly (Dec → Jan).
    ✅ Works with relative words like 'today', 'tomorrow', 'next week'.
    """
    if not text:
        return None

    text = text.strip().lower()
    today = datetime.now().date()
    current_year = today.year

    # --- Handle relative keywords ---
    if text in ["today", "now"]:
        return [today.isoformat()]
    if text == "tomorrow":
        return [(today + timedelta(days=1)).isoformat()]
    if "next week" in text:
        return [(today + timedelta(days=7)).isoformat()]

    # --- Detect explicit written dates (Nov, Dec, etc.) ---
    date_pattern = r"\b(\d{1,2}(?:st|nd|rd|th)?\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december))\b"
    matches = re.findall(date_pattern, text)
    normalized_dates = []

    for match in matches:
        try:
            # Start by parsing with current year
            parsed = parser.parse(f"{match} {current_year}", dayfirst=True).date()

            # If GPT gave old year (e.g., 2023), upgrade to current year
            if parsed.year < current_year:
                parsed = parsed.replace(year=current_year)

            # If the date has already passed this year, push to next year
            if parsed < today:
                parsed = parsed.replace(year=current_year + 1)

            normalized_dates.append(parsed.isoformat())
        except Exception as e:
            print(f"[WARN] normalize_date failed for '{match}': {e}")

    # --- Handle direct ISO dates like "2023-11-30" ---
    if not normalized_dates:
        try:
            parsed = datetime.fromisoformat(text).date()

            # Upgrade any old year to current year
            if parsed.year < current_year:
                parsed = parsed.replace(year=current_year)

            # Push to next year if already passed
            if parsed < today:
                parsed = parsed.replace(year=current_year + 1)

            normalized_dates = [parsed.isoformat()]
        except Exception:
            pass

    # --- Handle "return" keyword (auto second date if not present) ---
    if ("return" in text or "back" in text or "again" in text) and len(normalized_dates) == 1:
        try:
            dep_date = datetime.fromisoformat(normalized_dates[0])
            return_date = dep_date + timedelta(days=7)  # default return = +7 days
            normalized_dates.append(return_date.isoformat())
        except Exception as e:
            print(f"[WARN] could not auto-generate return date: {e}")

    # --- Cross-year fix for Dec → Jan returns ---
    if len(normalized_dates) > 1:
        dep = datetime.fromisoformat(normalized_dates[0])
        ret = datetime.fromisoformat(normalized_dates[1])
        if ret <= dep:
            ret = ret.replace(year=dep.year + 1)
            normalized_dates[1] = ret.isoformat()

    print(f"[DATE DEBUG] Input='{text}' → {normalized_dates}")
    return normalized_dates







# ==========================================================
#  FLIGHT PARAMETER BUILDER
# ==========================================================
DEFAULT_CURRENCY = "AED"

def build_flight_params(ctx):
    """Build Amadeus API search parameters safely."""
    if not ctx:
        ctx = {}

    # ✅ Ensure travel class and nonstop flags are valid
    travel_class = ctx.get("travelClass") or "ECONOMY"
    if isinstance(travel_class, str):
        travel_class = travel_class.upper()
    else:
        travel_class = "ECONOMY"

    nonstop = ctx.get("nonStop", False)
    if isinstance(nonstop, str):
        nonstop = nonstop.lower() in ["true", "yes", "1"]

    params = {
        "originLocationCode": ctx.get("origin"),
        "destinationLocationCode": ctx.get("destination"),
        "departureDate": ctx.get("departureDate"),
        "returnDate": ctx.get("returnDate"),
        "adults": int(ctx.get("adults", 1)),
        "children": int(ctx.get("children", 0)),
        "infants": int(ctx.get("infants", 0)),
        "travelClass": travel_class,
        "nonStop": str(nonstop).lower(),
        "currencyCode": ctx.get("currency", "AED"),
        "max": int(ctx.get("max", 10))
    }

    return params


def reset_stale_context():
    """
    Safely clears the user's session context and conversation memory.
    Used when context becomes inconsistent or too old.
    """
    global conversation_history
    session.pop("user_context", None)
    conversation_history = []
    print("[CONTEXT] 🔄 User session context has been reset.")

def get_preserved_outbound():
    """Fetch preserved outbound details from last_user_context.json, if available."""
    try:
        import json
        with open("last_user_context.json", "r") as f:
            saved = json.load(f)
            if all(k in saved for k in ["origin", "destination", "departureDate"]):
                return saved
    except Exception as e:
        print("[WARN] No preserved outbound context:", e)
    return None


def manage_context_reset(gpt_result):
    """🧠 Smart context manager — resets only when route changes."""
    import json, os
    try:
        # No previous context = nothing to reset
        if not os.path.exists("last_user_context.json"):
            print("[CONTEXT] ℹ️ No previous context found (fresh search).")
            return

        # Load previous context
        with open("last_user_context.json") as f:
            prev = json.load(f)

        prev_origin = (prev.get("origin") or "").lower()
        prev_dest = (prev.get("destination") or "").lower()
        new_origin = (gpt_result.get("origin") or "").lower()
        new_dest = (gpt_result.get("destination") or "").lower()

        # Compare routes
        if prev_origin != new_origin or prev_dest != new_dest:
            os.remove("last_user_context.json")
            print("[CONTEXT] 🧹 New route detected — old context cleared.")
        else:
            print("[CONTEXT] ♻️ Same route — keeping context for refinements.")

    except Exception as e:
        print(f"[WARN] Context management error: {e}")


# ===============================================================
# 🔹 Routes
# ===============================================================

@app.route("/api/search", methods=["POST"])
def search_flights():
    global conversation_history
    import json, re, os
    from datetime import datetime, timedelta
    from flask import request, jsonify

    data = request.get_json()
    query = data.get("query", "").strip()
    print(f"[USER QUERY] {query}")

    # =========================================================
    # ✅ STEP 1: Parse user input through GPT
    # =========================================================
    gpt_result = interpret_query_with_gpt(query)
    print("[GPT PARSED]", gpt_result)

    # =========================================================
    # ✅ STEP 2: Intelligent Context Reset
    # =========================================================
    manage_context_reset(gpt_result)

    # =========================================================
    # ✅ STEP 3: Dispatch to correct search handler
    # =========================================================
    intent = gpt_result.get("intent", "").lower()

    if intent == "search_multicity":
        return search_multicity_from_data(gpt_result)

    elif intent == "search_flights":
        return search_flights_from_data(gpt_result)

    else:
        # 💬 Handle non-flight / general chat queries
        general_intents = [
            "general_chat", "search_weather", "search_time",
            "search_hotels", "search_activities",
            "search_places", "search_general"
        ]

        if gpt_result.get("intent") in general_intents:
            # fallback AI response for general topics
            response_text = gpt_fallback_response(query)
            return jsonify({
                "type": "text",
                "text": response_text
            })

        # 🗣️ Small-talk / unknown fallback
        friendly_replies = {
            "hi": "👋 Hello there! How can I help you today?",
            "hello": "Hi! 😊 Looking to plan a trip or check flights?",
            "hey": "Hey! ✈️ Need help finding something?",
            "thanks": "You're welcome! Always happy to assist!",
            "thank you": "You're most welcome! 💙",
        }

        lower_query = query.lower().strip()
        if lower_query in friendly_replies:
            return jsonify({"type": "text", "text": friendly_replies[lower_query]})

        # If truly unrecognized, default assistant tone
        return jsonify({
            "type": "text",
            "text": "💬 Sure! Could you please clarify what you’d like me to help with?"
        })


    # =========================================================
    # 🧩 HELPERS
    # =========================================================
    def fmt_date(d):
        """Format date into human-readable '12 Nov 2025'."""
        try:
            return datetime.strptime(d.split("T")[0], "%Y-%m-%d").strftime("%d %b %Y")
        except Exception:
            return d

    def execute_single_flight_search(context, origin_code, dest_code, departure_date):
        """Execute a single Amadeus search and format results."""
        try:
            flight_params = build_flight_params(context)
            flight_params.update({
                "originLocationCode": origin_code,
                "destinationLocationCode": dest_code,
                "departureDate": departure_date.split("T")[0],
            })
            flight_params.pop("returnDate", None)

            print(f"[AMADEUS PARAMS - SINGLE LEG] {flight_params}")
            raw_response = query_amadeus_flights(flight_params)
            offers = raw_response.get("data", [])
            formatted = format_amadeus_response(raw_response, flight_params) if offers else []
            return formatted
        except Exception as e:
            print(f"[ERROR] Single-leg search failed: {e}")
            return []

    # =========================================================
    # 🧠 CONTEXT LOADING
    # =========================================================
    conversation_history.append({"role": "user", "content": query})
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    user_context = get_context() or {}
    if not user_context:
        try:
            with open("last_user_context.json", "r") as f:
                user_context = json.load(f)
        except Exception:
            user_context = {}

    # =========================================================
    # 🧭 GPT PARSING
    # =========================================================
    gpt_result = interpret_query_with_gpt(query, context=user_context)
    print("[GPT PARSED]", gpt_result)

    # =========================================================
    # 🗓️ Normalize and correct dates
    # =========================================================
    dates = normalize_date(query)
    if dates:
        dep = gpt_result.get("departureDate")
        if not dep or dep.startswith("2023") or dep < str(datetime.now().date()):
            gpt_result["departureDate"] = dates[0]
        if len(dates) > 1:
            gpt_result["returnDate"] = dates[1]

    # =========================================================
    # 🧠 CONTEXT MERGE LOGIC
    # =========================================================
    print(f"[CONTEXT BEFORE MERGE] origin={user_context.get('origin')}, dest={user_context.get('destination')}, dep_date={user_context.get('departureDate')}, ret_date={user_context.get('returnDate')}")

    if "return" in query.lower():
        print("[STRATEGY] Return trip detected - preserving original outbound")
        if user_context.get("departureDate"):
            gpt_result["departureDate"] = user_context["departureDate"]
        if user_context.get("origin"):
            gpt_result["origin"] = user_context["origin"]
        if user_context.get("destination"):
            gpt_result["destination"] = user_context["destination"]
        dates = normalize_date(query)
        if dates and len(dates) > 0:
            gpt_result["returnDate"] = dates[0]
            print(f"[RETURN DATE UPDATED] {gpt_result['returnDate']}")
    else:
        print("[STRATEGY] New search - normal context merge")
        if not gpt_result.get("origin") and user_context.get("origin"):
            gpt_result["origin"] = user_context["destination"]
        if not gpt_result.get("destination") and user_context.get("destination"):
            gpt_result["destination"] = user_context["origin"]
        if not gpt_result.get("departureDate") and user_context.get("departureDate"):
            gpt_result["departureDate"] = user_context["departureDate"]

    print(f"[CONTEXT AFTER MERGE] origin={gpt_result.get('origin')}, dest={gpt_result.get('destination')}, dep_date={gpt_result.get('departureDate')}, ret_date={gpt_result.get('returnDate')}")

    # ✅ Save merged context
    update_context(gpt_result)
    user_context = get_context()

    # =========================================================
    # ✈️ FLIGHT SEARCH HANDLING (One-way & Round-trip)
    # =========================================================
    if gpt_result.get("intent") == "search_flights":
        origin = gpt_result.get("origin")
        destination = gpt_result.get("destination")
        departure_date = gpt_result.get("departureDate")
        return_date = gpt_result.get("returnDate")

        context_origin = user_context.get("origin")
        context_destination = user_context.get("destination")

        if not context_origin or not context_destination:
            user_context["origin"] = origin
            user_context["destination"] = destination
            update_context(user_context)
            context_origin, context_destination = origin, destination

        outbound_origin = context_origin or origin
        outbound_dest = context_destination or destination

        print(f"[FINAL ROUTE] Using: {outbound_origin}→{outbound_dest}, Depart: {departure_date}, Return: {return_date}")

        # --- Round-trip ---
        if return_date:
            if not user_context.get("__locked_origin"):
                user_context["__locked_origin"] = outbound_origin
                user_context["__locked_dest"] = outbound_dest
                update_context(user_context)
                print(f"[LOCKED OUTBOUND] {outbound_origin}→{outbound_dest}")

            outbound_origin = user_context["__locked_origin"]
            outbound_dest = user_context["__locked_dest"]
            inbound_origin = outbound_dest
            inbound_dest = outbound_origin

            print(f"[ROUNDTRIP DEBUG] Outbound: {outbound_origin}→{outbound_dest} on {departure_date}, Return: {inbound_origin}→{inbound_dest} on {return_date}")

            outbound_results = execute_single_flight_search(user_context, outbound_origin, outbound_dest, departure_date)
            inbound_results = execute_single_flight_search(user_context, inbound_origin, inbound_dest, return_date)

            trip_type = "roundTrip"
            message = (
                f"Here are your flights from {outbound_origin} → {outbound_dest} "
                f"on {fmt_date(departure_date)} and back on {fmt_date(return_date)}:"
            )

            flights_data = [
                {
                    "type": "outbound",
                    "origin": outbound_origin,
                    "destination": outbound_dest,
                    "route": f"{outbound_origin} → {outbound_dest}",
                    "date": departure_date,
                    "offers": outbound_results
                },
                {
                    "type": "return",
                    "origin": inbound_origin,
                    "destination": inbound_dest,
                    "route": f"{inbound_origin} → {inbound_dest}",
                    "date": return_date,
                    "offers": inbound_results
                }
            ]

        # --- One-way ---
        else:
            if not user_context.get("__locked_origin"):
                user_context["__locked_origin"] = outbound_origin
                user_context["__locked_dest"] = outbound_dest
                update_context(user_context)
                print(f"[LOCKED OUTBOUND] {outbound_origin}→{outbound_dest}")

            outbound_results = execute_single_flight_search(user_context, outbound_origin, outbound_dest, departure_date)
            trip_type = "oneWay"
            message = f"Here are your flights from {outbound_origin} → {outbound_dest} on {fmt_date(departure_date)}:"
            flights_data = [
                {
                    "type": "outbound",
                    "origin": outbound_origin,
                    "destination": outbound_dest,
                    "route": f"{outbound_origin} → {outbound_dest}",
                    "date": departure_date,
                    "offers": outbound_results
                }
            ]

        # ✅ Return response for search_flights
        return jsonify({
            "type": "flight_recommendation",
            "text": message,
            "data": {"tripType": trip_type, "flights": flights_data}
        })

    # =========================================================
    # ✈️ MULTI-CITY HANDLING
    # =========================================================
    if gpt_result.get("intent") == "search_multicity":
        print("[HANDLER] ✈️ Multi-city intent detected — resetting context before search.")
        try:
            reset_stale_context()
            if os.path.exists("last_user_context.json"):
                os.remove("last_user_context.json")
                print("[CONTEXT] 🧹 Cleared old last_user_context.json for fresh multi-city search.")
        except Exception as e:
            print(f"[WARN] Could not reset multi-city context: {e}")
        return search_multicity_from_data(gpt_result)


    # =========================================================
    # 💬 GENERAL CHAT HANDLING
    # =========================================================
    if gpt_result.get("intent") == "general_chat":
        text = gpt_result.get("text") or "Hello! How can I assist you today?"
        print(f"[GENERAL CHAT] {text}")
        return jsonify({"type": "text", "text": text})


    # =========================================================
    # 🤖 GPT-POWERED GENERAL RESPONSE HANDLER
    # =========================================================
    if gpt_result.get("intent") in [
        "search_weather",
        "search_time",
        "search_hotels",
        "search_activities",
        "search_places",
        "search_general"
    ]:
        user_query = query
        print(f"[GPT-FALLBACK] Handling {gpt_result.get('intent')} via GPT: {user_query}")
        ai_text = gpt_fallback_response(user_query)
        return jsonify({"type": "text", "text": ai_text})


    # =========================================================
    # 🪶 DEFAULT FALLBACK
    # =========================================================
    return jsonify({"type": "text", "text": "Unrecognized or general query."})












def general_chat_fallback(history):
    """
    Smart fallback handler for general chat or non-travel messages.
    Uses OpenAI GPT to respond naturally while staying travel-focused.
    """
    try:
        user_message = history[-1]['content'] if history else "Hello"

        # Prepare conversation context for GPT
        messages = [
            {"role": "system", "content": (
                "You are a friendly AI travel assistant. "
                "If the user greets you or chats casually, reply warmly and briefly. "
                "If they ask about travel, respond helpfully. "
                "Keep your tone concise, friendly, and relevant to travel planning."
            )},
            {"role": "user", "content": user_message},
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.8,
            max_tokens=120,
        )

        reply = completion.choices[0].message.content.strip()
        print(f"[GPT SMALL-TALK REPLY] {reply}")

        return jsonify({"type": "text", "text": reply, "data": None})

    except Exception as e:
        print(f"[ERROR] general_chat_fallback failed: {e}")
        return jsonify({
            "type": "text",
            "text": "💬 Hi there! I can help you find flights, check weather, or tell you local time. What would you like to do?",
            "data": None
        })





#######################################
## MULTICITY##
#######################################
def search_flights_from_data(gpt_result):
    """
    ✅ FINAL — Smart Flight Search (AED + One-Way + Round-Trip)
    -----------------------------------------------------------
    • Detects one-way vs round-trip automatically
    • Ensures Jan after Nov → next year (2026 logic)
    • Past dates corrected to next valid future year
    • Always returns fares in AED
    • Handles 400 / 429 API errors gracefully
    """
    import json, time
    from datetime import datetime, date
    from flask import jsonify

    print("[FLIGHT SEARCH] Running Amadeus query (AED locked)...")

    # --- Passenger & class info ---
    pax_adults = int(gpt_result.get("adults", 1))
    pax_children = int(gpt_result.get("children", 0))
    pax_infants = int(gpt_result.get("infants", 0))
    travel_class = (gpt_result.get("travelClass") or "ECONOMY").upper()
    if "PREMIUM" in travel_class and "ECONOMY" not in travel_class:
        travel_class = "PREMIUM_ECONOMY"
    nonstop = str(gpt_result.get("nonStop")).lower() in ["true", "1", "yes"]
    currency = "AED"  # 🔒 Always AED

    origin_raw = (gpt_result.get("origin") or "").strip()
    dest_raw = (gpt_result.get("destination") or "").strip()
    dep_date_raw = gpt_result.get("departureDate")
    ret_date_raw = gpt_result.get("returnDate")

    today = date.today()

    # --- Smart date correction ---
    def normalize_date(d_raw):
        try:
            d = datetime.fromisoformat(str(d_raw)).date()
        except Exception:
            try:
                d = datetime.strptime(str(d_raw).split("T")[0], "%Y-%m-%d").date()
            except Exception:
                d = today
        if d < today:
            d = d.replace(year=today.year)
            if d < today:
                d = d.replace(year=today.year + 1)
        return d

    dep_date = normalize_date(dep_date_raw)
    ret_date = normalize_date(ret_date_raw) if ret_date_raw else None

    # --- Fix cross-year order (Jan after Nov → next year) ---
    if ret_date and ret_date <= dep_date:
        print(f"[ADJUST] Return bumped to next year to stay after departure.")
        ret_date = ret_date.replace(year=dep_date.year + 1)

    # --- Format date for print/UI ---
    def fmt_date(d):
        try:
            return datetime.strptime(str(d), "%Y-%m-%d").strftime("%d %b %Y")
        except Exception:
            return str(d)

    # --- Normalize IATA codes ---
    origin_code = get_iata_code(origin_raw) or origin_raw[:3].upper() or "UNK"
    dest_code = get_iata_code(dest_raw) or dest_raw[:3].upper() or "UNK"
    print(f"[LOOKUP] {origin_raw} → {origin_code}, {dest_raw} → {dest_code}")

    # --- Query helper ---
    def fetch_flights(origin, dest, d_obj):
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": dest,
            "departureDate": d_obj.isoformat(),
            "adults": pax_adults,
            "children": pax_children,
            "infants": pax_infants,
            "travelClass": travel_class,
            "currencyCode": currency,
            "max": 10
        }
        if nonstop:
            params["nonStop"] = True
        print(f"[AMADEUS] Query {origin}→{dest} on {d_obj.isoformat()} nonstop={nonstop}")
        try:
            raw = query_amadeus_flights(params)
            offers = raw.get("data", [])
            if not offers and nonstop:
                print("[RETRY] No nonstop results — retrying with stopovers.")
                params.pop("nonStop", None)
                raw = query_amadeus_flights(params)
                offers = raw.get("data", [])
            return format_amadeus_response(raw, params) if offers else []
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                print("[RATE LIMIT] 429 — retrying in 3s.")
                time.sleep(3)
                return fetch_flights(origin, dest, d_obj)
            elif "400" in msg and nonstop:
                print("[RETRY] 400 error — retrying with stopovers.")
                params.pop("nonStop", None)
                return fetch_flights(origin, dest, d_obj)
            print(f"[ERROR] Amadeus query failed: {e}")
            return []

    # =====================================================
    # ✈️ Handle One-Way and Round-Trip Separately
    # =====================================================
    outbound_flights = fetch_flights(origin_code, dest_code, dep_date)
    return_flights = []

    if ret_date:
        return_flights = fetch_flights(dest_code, origin_code, ret_date)

    # --- Safe fallback if no data ---
    def safe_no_results(o, d):
        return [{
            "airline": "N/A",
            "price": "N/A",
            "departure": o,
            "arrival": d,
            "duration": "—",
            "stops": 0,
            "message": "No flights found."
        }]

    if not outbound_flights:
        outbound_flights = safe_no_results(origin_code, dest_code)
    if ret_date and not return_flights:
        return_flights = safe_no_results(dest_code, origin_code)

    # =====================================================
    # 🧭 Build Tabs for UI
    # =====================================================
    flight_results = [{
        "tab": f"{origin_code} → {dest_code} ({fmt_date(dep_date)})",
        "origin": origin_code,
        "destination": dest_code,
        "departureDate": dep_date.isoformat(),
        "offers": outbound_flights
    }]

    if ret_date:
        flight_results.append({
            "tab": f"{dest_code} → {origin_code} ({fmt_date(ret_date)})",
            "origin": dest_code,
            "destination": origin_code,
            "departureDate": ret_date.isoformat(),
            "offers": return_flights
        })

    # =====================================================
    # 📦 Final Payload
    # =====================================================
    final_payload = {
        "flights": flight_results,
        "segments": flight_results,
        "outbound": outbound_flights,
        "inbound": return_flights
    }

    # =====================================================
    # 💾 Save Context
    # =====================================================
    try:
        gpt_result["intent"] = "search_flights"
        gpt_result["segments"] = flight_results
        with open("last_user_context.json", "w") as f:
            json.dump(gpt_result, f, indent=2)
        print("[INFO] Saved flight search context ✅")
    except Exception as e:
        print(f"[WARN] Could not save context: {e}")

    pax_text = f"{pax_adults} adult{'s' if pax_adults > 1 else ''}"
    if pax_children:
        pax_text += f", {pax_children} child{'ren' if pax_children > 1 else ''}"
    if pax_infants:
        pax_text += f", {pax_infants} infant{'s' if pax_infants > 1 else ''}"

    if ret_date:
        msg = (
            f"Here are your {travel_class.title()} fares in {currency} for {pax_text}: "
            f"{origin_code} → {dest_code} on {fmt_date(dep_date)} and back on {fmt_date(ret_date)}."
        )
    else:
        msg = (
            f"Here are your one-way {travel_class.title()} fares in {currency} "
            f"from {origin_code} → {dest_code} on {fmt_date(dep_date)} for {pax_text}."
        )

    print("[FLIGHT SEARCH DONE ✅]")
    return jsonify({
        "type": "flight_recommendation",
        "text": msg,
        "data": final_payload
    })






@app.route("/api/flow", methods=["POST"])
def handle_flow():
    data = request.get_json()
    selected = data.get("selected")

    if selected == "book_flight":
        conversation_state["stage"] = "awaiting_origin"
        return jsonify({
            "type": "text",
            "text": "✈️ Great! Where will you be flying from?",
            "data": None
        })

    elif selected == "book_hotel":
        conversation_state["stage"] = None
        return jsonify({
            "type": "text",
            "text": HOTEL_COMING_SOON_TEXT,
            "data": None
        })

    else:
        return jsonify({
            "type": "text",
            "text": "🤔 Sorry, I didn’t understand that option.",
            "data": None
        })

def search_multicity_from_data(gpt_result):
    """
    ✅ FINAL — Multi-City Flight Search (AED + Smart Dates)
    -------------------------------------------------------
    • Always returns fares in AED
    • Fixes past and cross-year dates (Nov 2025 → Jan 2026)
    • Keeps segments chronologically ordered
    • Handles Amadeus 400/429 gracefully
    """
    import json, time
    from datetime import datetime, date
    from flask import jsonify

    print("[MULTI-CITY] Running multi-city search via Amadeus (AED locked)...")

    # --- Passenger & class info ---
    pax_adults = int(gpt_result.get("adults", 1))
    pax_children = int(gpt_result.get("children", 0))
    pax_infants = int(gpt_result.get("infants", 0))
    travel_class = (gpt_result.get("travelClass") or "ECONOMY").upper()
    if "PREMIUM" in travel_class and "ECONOMY" not in travel_class:
        travel_class = "PREMIUM_ECONOMY"
    nonstop = str(gpt_result.get("nonStop")).lower() in ["true", "1", "yes"]
    currency = "AED"  # 🔒 always AED

    segments = gpt_result.get("segments", [])
    if not segments:
        return jsonify({
            "type": "text",
            "text": "⚠️ Please specify at least two cities for a multi-city trip.",
            "data": None
        })

    today = date.today()
    all_tabs, flight_results = [], []

    def fmt_date(d):
        try:
            return datetime.strptime(str(d).split("T")[0], "%Y-%m-%d").strftime("%d %b %Y")
        except Exception:
            return "Invalid Date"

    # =====================================================
    # 🛫 Process Each Segment
    # =====================================================
    for i, seg in enumerate(segments, start=1):
        origin_raw = (seg.get("origin") or "").strip()
        dest_raw = (seg.get("destination") or "").strip()
        dep_date_raw = seg.get("departureDate")

        origin_code = get_iata_code(origin_raw) or origin_raw.upper()[:3] or "UNK"
        dest_code = get_iata_code(dest_raw) or dest_raw.upper()[:3] or "UNK"

        if not origin_code or not dest_code or origin_code == dest_code:
            print(f"[WARN] Skipping invalid segment {i}: {origin_raw} → {dest_raw}")
            continue

        # =====================================================
        # 🧭 SMART DATE FIX (future-aware + chronological)
        # =====================================================
        try:
            d = datetime.fromisoformat(str(dep_date_raw)).date()
        except Exception:
            try:
                d = datetime.strptime(str(dep_date_raw).split("T")[0], "%Y-%m-%d").date()
            except Exception:
                d = today

        # --- MAIN YEAR CORRECTION ---
        if d < today:
            d = d.replace(year=today.year)
            if d < today:
                d = d.replace(year=today.year + 1)

        # --- KEEP CHRONOLOGICAL ORDER ACROSS SEGMENTS ---
        if flight_results:
            prev_date = datetime.fromisoformat(flight_results[-1]["departureDate"]).date()
            if d < prev_date:
                print(f"[ADJUST] {origin_code}→{dest_code} bumped to next year to stay after previous segment.")
                d = d.replace(year=d.year + 1)

        dep_date = d.isoformat()
        safe_date = fmt_date(dep_date)
        print(f"[DATE FIX] {origin_code}→{dest_code} → {dep_date}")

        # =====================================================
        # 🧠 Build Amadeus Params
        # =====================================================
        flight_params = {
            "originLocationCode": origin_code,
            "destinationLocationCode": dest_code,
            "departureDate": dep_date,
            "adults": pax_adults,
            "children": pax_children,
            "infants": pax_infants,
            "travelClass": travel_class,
            "currencyCode": currency,
            "max": 10
        }
        if nonstop:
            flight_params["nonStop"] = True

        print(f"[AMADEUS PARAMS - MULTI-CITY SEGMENT {i}] {flight_params}")

        # =====================================================
        # 🚀 Query Amadeus API (Retry on 400/429)
        # =====================================================
        formatted = []
        try:
            raw_response = query_amadeus_flights(flight_params)
            offers = raw_response.get("data", [])
            if not offers and nonstop:
                print("[RETRY] No nonstop flights — retrying with stopovers.")
                flight_params.pop("nonStop", None)
                raw_response = query_amadeus_flights(flight_params)
                offers = raw_response.get("data", [])
            formatted = format_amadeus_response(raw_response, flight_params) if offers else []
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                print("[RATE LIMIT] 429 — retrying in 3s.")
                time.sleep(3)
                try:
                    raw_response = query_amadeus_flights(flight_params)
                    offers = raw_response.get("data", [])
                    formatted = format_amadeus_response(raw_response, flight_params) if offers else []
                except Exception as e2:
                    print("[FAIL] Retry failed:", e2)
            elif "400" in msg and nonstop:
                print("[RETRY] 400 error with nonstop=True — retrying with stopovers.")
                flight_params.pop("nonStop", None)
                try:
                    raw_response = query_amadeus_flights(flight_params)
                    offers = raw_response.get("data", [])
                    formatted = format_amadeus_response(raw_response, flight_params) if offers else []
                except Exception as e3:
                    print("[FAIL] Second retry failed:", e3)
            else:
                print(f"[ERROR] Amadeus query failed for segment {i}: {e}")
                formatted = []

        # --- Fallback if empty ---
        if not formatted:
            formatted = [{
                "airline": "N/A",
                "price": "N/A",
                "departure": origin_code,
                "arrival": dest_code,
                "duration": "—",
                "stops": 0,
                "message": "No flights found."
            }]

        tab_label = f"{origin_code} → {dest_code} ({safe_date})"
        print(f"[MULTI-CITY SEGMENT {i}] {tab_label} → {len(formatted)} flights found.")

        all_tabs.append({
            "tab": tab_label,
            "origin": origin_code,
            "destination": dest_code,
            "departureDate": dep_date,
            "offers": formatted
        })
        flight_results.append({
            "segment": i,
            "origin": origin_code,
            "destination": dest_code,
            "departureDate": dep_date,
            "offers": formatted
        })

    # =====================================================
    # 🧩 Chronological Order for UI Tabs
    # =====================================================
    all_tabs.sort(key=lambda x: x["departureDate"])
    flight_results.sort(key=lambda x: x["departureDate"])

    # =====================================================
    # 📦 Final JSON Payload
    # =====================================================
    final_payload = {
        "flights": all_tabs,
        "segments": flight_results,
        "outbound": flight_results[0]["offers"] if flight_results else [],
        "inbound": []
    }

    # =====================================================
    # 💾 Save Context for Continuity
    # =====================================================
    try:
        gpt_result["intent"] = "search_multicity"
        gpt_result["segments"] = flight_results
        with open("last_user_context.json", "w") as f:
            json.dump(gpt_result, f, indent=2)
        print("[INFO] Saved multi-city context ✅")
    except Exception as e:
        print(f"[WARN] Could not save multi-city context: {e}")

    # =====================================================
    # 🧾 Passenger Summary
    # =====================================================
    pax_text = f"{pax_adults} adult{'s' if pax_adults > 1 else ''}"
    if pax_children:
        pax_text += f", {pax_children} child{'ren' if pax_children > 1 else ''}"
    if pax_infants:
        pax_text += f", {pax_infants} infant{'s' if pax_infants > 1 else ''}"

    msg = f"Here are your multi-city {travel_class.title()} fares in {currency} for {pax_text}:"
    print("[MULTI-CITY DONE ✅] Generated", len(all_tabs), "tabs successfully.")

    # =====================================================
    # ⚠️ No Results Fallback
    # =====================================================
    if all(len(seg["offers"]) == 1 and seg["offers"][0].get("airline") == "N/A" for seg in all_tabs):
        msg = "⚠️ No flights could be found for your multi-city itinerary. Try adjusting your dates or routes."

    return jsonify({
        "type": "flight_recommendation",
        "text": msg,
        "data": final_payload
    })



# ===============================================================
# 🔹 Run Server
# ===============================================================
from flask import send_from_directory

@app.route("/")
def serve_index():
    """Serve the frontend index.html file when visiting the root URL."""
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

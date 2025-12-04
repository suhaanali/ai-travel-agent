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

hotel_image_cache = {}
EXPEDIA_REGION_MAP = {
    "dubai": "2998",
    "abu dhabi": "2948",
    "sharjah": "3007",
    "ras al khaimah": "2995",
    "fujairah": "2996",
    "london": "178279",
    "paris": "179898",
    "new york": "178293",
    "singapore": "172674",
    "tokyo": "179900",
}

PROPERTY_IDS = [
    6394, 21870, 910165, 1001421, 114730, 15291, 190827,
    517439, 81083, 906456, 114530, 115885, 985916, 115888,
    432678, 527498, 527497, 527499, 486304, 47377, 1545207,
    903302, 48995, 559080, 804814, 863980, 850069, 521243,
    83995, 869326, 891770, 4631636
]
print(">>> RUNNING FROM FILE:", __file__)

# ==========================================================
#  FIRST: Load ENV and Expedia credentials
# ==========================================================

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

EXPEDIA_KEY = os.getenv("EXPRapidApiKey")
EXPEDIA_SECRET = os.getenv("EXPRapidSecret")
EXPEDIA_API_URL = os.getenv("EXPRapidApiUrl")

# Debug prints (AFTER loading)
print("Loaded Expedia Key =", EXPEDIA_KEY)
print("Loaded Expedia Secret =", EXPEDIA_SECRET)
print("Loaded Expedia URL =", EXPEDIA_API_URL)

# ==========================================================
#  Flask initialization
# ==========================================================

flask_key = os.environ.get("FLASK_SECRET_KEY")
app = Flask(__name__)
CORS(app, supports_credentials=True)


# --- Set secret key ---
if flask_key:
    app.secret_key = flask_key
    print(f"[INFO] ✅ FLASK_SECRET_KEY loaded from .env ({len(flask_key)} chars)")
else:
    print("[WARN] ⚠️ No FLASK_SECRET_KEY found — generating temporary key.")
    app.secret_key = os.urandom(24)

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

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    print(f"[DEBUG] Fetching weather for {city} → {url}")

    # Weather emoji map
    WEATHER_ICONS = {
        "clear": "☀️",
        "clouds": "☁️",
        "rain": "🌧️",
        "thunderstorm": "⛈️",
        "drizzle": "🌦️",
        "snow": "❄️",
        "mist": "🌫️",
        "fog": "🌫️",
        "haze": "🌫️",
        "smoke": "🌫️",
        "dust": "🌫️",
        "sand": "🌫️",
        "ash": "🌫️",
        "squall": "💨",
        "tornado": "🌪️"
    }

    try:
        response = requests.get(url, timeout=10)
        print(f"[DEBUG] Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            # extract weather info
            main_weather = data["weather"][0]["main"].lower()
            desc = data["weather"][0]["description"].capitalize()
            temp = round(data["main"]["temp"])
            feels = round(data["main"]["feels_like"])
            humidity = data["main"]["humidity"]

            # pick emoji
            icon = WEATHER_ICONS.get(main_weather, "🌤️")

            # return formatted weather
            return f"{icon} {desc}, {temp}°C (feels like {feels}°C, humidity {humidity}%)"

        if response.status_code == 404:
            return f"⚠️ City '{city}' not found. Please check the spelling."

        return f"⚠️ Weather service returned error {response.status_code}."

    except Exception as e:
        print("[ERROR] Weather fetch failed:", e)
        return "⚠️ Could not fetch weather right now."

def get_weather_forecast(city, days=3):
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    if not API_KEY:
        return "⚠️ Weather API key not configured."

    # OpenWeather 3-hour forecast (40 entries = 5 days)
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"⚠️ Could not fetch forecast for {city}."

        data = response.json()

        # Group forecast by day
        from collections import defaultdict
        groups = defaultdict(list)

        for item in data["list"]:
            date = item["dt_txt"].split(" ")[0]
            groups[date].append(item)

        forecast_texts = []
        count = 0

        for date, items in groups.items():
            if count >= days:
                break

            # Take middle-of-day forecast for readability
            mid = items[len(items)//2]
            desc = mid["weather"][0]["description"].capitalize()
            temp = mid["main"]["temp"]
            feels = mid["main"]["feels_like"]
            humidity = mid["main"]["humidity"]

            # Icon (simple — emoji version)
            icon_map = {
                "clear": "☀️",
                "cloud": "🌤️",
                "rain": "🌧️",
                "thunder": "⛈️",
                "snow": "❄️"
            }
            icon = "🌦️"
            for key, val in icon_map.items():
                if key in desc.lower():
                    icon = val
                    break

            forecast_texts.append(
                f"{icon} **{date}** — {desc}, {temp}°C (feels like {feels}°C, humidity {humidity}%)"
            )
            count += 1

        return "\n".join(forecast_texts)

    except Exception as e:
        print("[ERROR] Forecast fetch failed:", e)
        return f"⚠️ Could not fetch forecast data."




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

    # ----------------------------------------------------
    # ALWAYS normalize multi-city dates (fix inconsistent behavior)
    # ----------------------------------------------------
    if parsed.get("intent") == "search_multicity" and parsed.get("segments"):
        for seg in parsed["segments"]:
            if seg.get("departureDate"):
                fixed = normalize_future_date(seg["departureDate"])
                seg["departureDate"] = fixed.isoformat()
    # ----------------------------------------------------

    # 1️⃣ If this is a follow-up like "return on 15 nov"
    if "return" in text and re.search(r"\d{1,2}\s*[a-zA-Z]{3,}", text):
        match = re.search(r"(\d{1,2}\s*[a-zA-Z]{3,})", text)
        if match:
            parsed_date = datetime.strptime(match.group(1) + f" {datetime.now().year}", "%d %b %Y")
            user_context["returnDate"] = parsed_date.strftime("%Y-%m-%d")

        if user_context.get("origin") and user_context.get("destination"):
            origin, destination = user_context["origin"], user_context["destination"]
            user_context["origin"], user_context["destination"] = destination, origin

        parsed.update(user_context)
        return build_flight_query(parsed)

    # 2️⃣ Relative date adjustments
    if re.search(r"next day|previous day|days? (later|after|before|earlier)", text):
        user_context = adjust_relative_date(user_input, user_context)
        parsed.update(user_context)
        return build_flight_query(parsed)

    # 3️⃣ Save new info to context
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
    Ultra-Smart Travel Interpreter v5
    Handles: flights, multicity, hotels, weather, time, etc.
    Returns clean JSON ONLY.
    """
    from datetime import datetime, timedelta
    import json, re

    system_prompt = """
You are an expert multilingual travel planning AI that converts user travel queries into structured JSON for APIs.

Your responsibilities:
- Detect the user's true travel intent (flights, multi-city, hotels, weather, time, etc.)
- Understand natural phrases, dates, ranges, airports, cities, and constraints.
- Output ONLY valid JSON. Never output text, markdown, comments, or explanations.

----------------------------------------------------
🧭 RECOGNIZED INTENTS
----------------------------------------------------
- "search_flights"
- "search_multicity"
- "search_hotels"
- "search_weather"
- "search_time"
- "search_general"
- "general_chat"
- "search_currency"

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
  "preferredAirlines": ["<airlines>"],
  "alliances": ["<Star Alliance|Oneworld|SkyTeam>"],
  "maxStops": "<int or null>",
  "text": "<brief English summary>"
}

----------------------------------------------------
🏨 HOTEL SEARCH RULES
----------------------------------------------------
Hotel JSON MUST be:
{
  "intent": "search_hotels",
  "city": "<city>",
  "checkIn": "<YYYY-MM-DD>",
  "checkOut": "<YYYY-MM-DD>",
  "nights": <int>,
  "adults": <int>,
  "rooms": <int>,
  "text": "<summary>"
}

Hotel rules:
- Never use origin/destination/departureDate/returnDate for hotels.
- If user says “check in 19 Nov for 3 nights”
    - checkIn = parsed date
    - checkOut = +3 nights
    - nights = 3
- If date range “19–22 Nov”
    - nights = difference
- Defaults: adults=1, rooms=1

----------------------------------------------------
✈️ FLIGHT RULES
----------------------------------------------------
(unchanged)

----------------------------------------------------
⛔ OUTPUT RULES
----------------------------------------------------
Always output ONLY valid JSON.
"""

    try:
        messages = [{"role": "system", "content": system_prompt}]

        # Context
        if context:
            messages.append({"role": "system",
                             "content": f"Previous context: {json.dumps(context, ensure_ascii=False)}"})

        messages.append({"role": "user", "content": query})

        # Call GPT
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(completion.choices[0].message.content)

        # ===============================
        # POST PROCESSING FIXES
        # ===============================
        today = datetime.now().date()

        # --- Date range parser
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

        # ===============================
        # ✈️ FLIGHT DATE LOGIC ONLY
        # ===============================
        if parsed.get("intent") in ["search_flights", "search_multicity"]:
            dep_start, dep_end = parse_date_range(query)

            if "return" in query.lower():
                ret_start, ret_end = parse_date_range(query.split("return")[-1])
            else:
                ret_start, ret_end = (None, None)

            if dep_start and not parsed.get("departureDate"):
                parsed["departureDate"] = dep_start

            if ret_end and not parsed.get("returnDate"):
                parsed["returnDate"] = ret_end

        # ===============================
        # MULTICITY SEGMENT AUTO-BUILD
        # ===============================
        if parsed.get("intent") == "search_multicity" and not parsed.get("segments"):
            via_match = re.findall(r"via\s+([\w\s,]+)", query.lower())
            via_cities = []

            if via_match:
                via_cities = [x.strip().title() for x in re.split(r",|or|and", via_match[0]) if x.strip()]

            origin = parsed.get("origin")
            destination = parsed.get("destination")
            if origin and destination:
                chain = [origin] + via_cities + [destination]
                parsed["segments"] = [
                    {"origin": chain[i], "destination": chain[i + 1],
                     "departureDate": parsed.get("departureDate")}
                    for i in range(len(chain) - 1)
                ]

        # ===============================
        # AUTO RETURN FLIGHT CREATION
        # ===============================
        if parsed.get("intent") == "search_flights" and not parsed.get("segments"):
            if parsed.get("origin") and parsed.get("destination"):
                parsed["segments"] = [{
                    "origin": parsed["origin"],
                    "destination": parsed["destination"],
                    "departureDate": parsed.get("departureDate")
                }]
                if parsed.get("returnDate"):
                    parsed["segments"].append({
                        "origin": parsed["destination"],
                        "destination": parsed["origin"],
                        "departureDate": parsed["returnDate"]
                    })

        # ===============================
        # NORMALIZE CLASS
        # ===============================
        if parsed.get("travelClass"):
            cls = parsed["travelClass"].upper().replace(" ", "_")
            if "PREMIUM" in cls and "ECONOMY" not in cls:
                cls = "PREMIUM_ECONOMY"
            parsed["travelClass"] = cls

        # ===============================
        # SAFE DEFAULTS (FLIGHT ONLY)
        # ===============================
        if parsed.get("intent") in ["search_flights", "search_multicity"]:
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


def get_context():
    """Retrieve the current user's context from their specific browser session."""
    if "user_context" not in session:
        session["user_context"] = {}
    return session["user_context"]

def update_context(new_data):
    """Update context only for this specific user."""
    ctx = session.get("user_context", {})
    
    # Merge new data into existing context
    for k, v in new_data.items():
        if v: # Only update if value is not None/Empty
            ctx[k] = v
            
    session["user_context"] = ctx
    session.modified = True # Tell Flask to save the session
    return ctx

def reset_stale_context():
    """Clear context for this user only."""
    session.pop("user_context", None)
    print("[CONTEXT] 🔄 Session context cleared for current user.")




# ===============================================================
# 🔹 Routes
# ===============================================================

@app.route("/api/search", methods=["POST"])
def search_flights():
    global conversation_history
    import json, re, os, requests
    from datetime import datetime, timedelta
    from flask import request, jsonify

    data = request.get_json()
    query = data.get("query", "").strip()
    print(f"[USER QUERY] {query}")

    # STEP 1: Parse with GPT
    user_context = get_context() or {}
    gpt_result = interpret_query_with_gpt(query, context=user_context)
    print("[GPT PARSED]", gpt_result)

    # STEP 3: Dispatch
    intent = gpt_result.get("intent", "").lower()

    # ✈️ Multi-city flights
    if intent == "search_multicity":
        return search_multicity_from_data(gpt_result)

    # ✈️ Regular flights
    elif intent == "search_flights":
        return search_flights_from_data(gpt_result)

    # 🏨 Hotels
    elif intent == "search_hotels":
        print("[ROUTER] Forwarding to LiteAPI with:", gpt_result)
        return api_search_hotels(data=gpt_result)

    # 🌦 Weather
    elif intent == "search_weather":
        city = gpt_result.get("destination") or gpt_result.get("origin")

        # Detect “next X days”
        days = 1
        match_days = re.search(r"next\s+(\d+)\s+days?", query.lower())
        if match_days:
            days = int(match_days.group(1))

        # Regex fallback
        if not city:
            match = re.search(r"(?:in|for|at)\s+([a-zA-Z\s]+)$", query.lower())
            if match:
                city = match.group(1).strip()

        if not city:
            ctx = get_context() or {}
            city = ctx.get("destination")

        if not city:
            return jsonify({"type": "text", "text": "⚠️ I couldn't detect the city."})

        # Forecast
        if days > 1:
            wx = get_weather_forecast(city, days)
            return jsonify({"type": "text", "text": wx})

        # Current weather
        wx = get_weather(city)
        return jsonify({"type": "text", "text": wx})

    # 🕒 Time handler
    elif intent == "search_time":
        city = gpt_result.get("destination") or gpt_result.get("origin")

        # If user asks “what is the date today?”
        if not city and "date" in query.lower():
            today = datetime.now().strftime("%A, %d %B %Y")
            return jsonify({"type": "text", "text": f"📅 Today is {today}."})

        # Regex fallback
        if not city:
            match = re.search(r"in\s+([a-zA-Z\s]+)$", query.lower())
            if match:
                city = match.group(1).strip()

        if not city:
            return jsonify({"type": "text", "text": "⚠️ Please specify a city."})

        result = get_time(city)
        return jsonify({"type": "text", "text": result})

    # 💱 Currency Converter
    elif intent == "search_currency":

        # Use GPT-parsed values first
        amount = gpt_result.get("amount")
        from_cur = gpt_result.get("from")
        to_cur = gpt_result.get("to")

        text = query.lower()

        # Regex fallback for numbers + 3-letter currencies
        if not amount:
            amt = re.search(r"(\d+\.?\d*)", text)
            if amt:
                amount = float(amt.group(1))

        if not from_cur or not to_cur:
            cur = re.findall(r"\b[a-zA-Z]{3}\b", text.upper())
            if len(cur) >= 2:
                from_cur = from_cur or cur[0]
                to_cur = to_cur or cur[1]

        # Validation
        if not amount or not from_cur or not to_cur:
            return jsonify({
                "type": "text",
                "text": "⚠️ Please say: convert 100 AED to INR"
            })

        # API call
        api_key = "5ce955cfa68b6faaa9542603"
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{from_cur}"

        try:
            res = requests.get(url, timeout=10).json()

            if res.get("result") != "success":
                return jsonify({"type": "text",
                                "text": f"⚠️ API error: {res.get('error-type')}"})

            rate = res["conversion_rates"][to_cur]
            converted = round(amount * rate, 2)

            return jsonify({
                "type": "text",
                "text": f"💱 {amount} {from_cur} = {converted} {to_cur}"
            })

        except Exception as e:
            print("FX error:", e)
            return jsonify({"type": "text",
                            "text": "⚠️ Could not reach currency API."})

    # 💬 Fallback for everything else
    else:
        response_text = gpt_fallback_response(query)
        return jsonify({"type": "text", "text": response_text})








# =========================================================
# 🧩 HELPERS
# =========================================================
def fmt_date(d):
    """Format date into '12 Nov 2025'."""
    from datetime import datetime
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
            # ✅ ONLY clear the session for this specific user
            reset_stale_context()
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
    Unified search engine for One-Way + Round-Trip + Multi-City
    """
    import json, time
    from datetime import datetime, date
    from flask import jsonify

    print("[FLIGHT SEARCH] Running unified Amadeus search (AED locked)...")

    # --- Passenger info ---
    pax_adults = int(gpt_result.get("adults", 1))
    pax_children = int(gpt_result.get("children", 0))
    pax_infants = int(gpt_result.get("infants", 0))

    travel_class = (gpt_result.get("travelClass") or "ECONOMY").upper()
    if "PREMIUM" in travel_class and "ECONOMY" not in travel_class:
        travel_class = "PREMIUM_ECONOMY"

    nonstop = str(gpt_result.get("nonStop")).lower() in ["true", "yes", "1"]
    currency = "AED"

    origin_raw = (gpt_result.get("origin") or "").strip()
    dest_raw = (gpt_result.get("destination") or "").strip()
    dep_date_raw = gpt_result.get("departureDate")
    ret_date_raw = gpt_result.get("returnDate")

    today = date.today()

    # --- Date Normalizer ---
    def normalize_date(d_raw):
        try:
            d = datetime.fromisoformat(str(d_raw)).date()
        except:
            try:
                d = datetime.strptime(str(d_raw).split("T")[0], "%Y-%m-%d").date()
            except:
                d = today

        if d < today:
            d = d.replace(year=today.year)
            if d < today:
                d = d.replace(year=today.year + 1)

        return d

    dep_date = normalize_date(dep_date_raw)
    ret_date = normalize_date(ret_date_raw) if ret_date_raw else None

    # --- Fix only true Dec→Jan return ---
    if ret_date and ret_date < dep_date:
        if dep_date.month == 12 and ret_date.month == 1:
            print("[CROSS-YEAR] Valid Dec→Jan return")
        else:
            print("[FIX] Return earlier → forcing same year")
            ret_date = ret_date.replace(year=dep_date.year)

    # --- IATA codes ---
    origin_code = get_iata_code(origin_raw) or origin_raw[:3].upper()
    dest_code = get_iata_code(dest_raw) or dest_raw[:3].upper()

    # --- Build unified segments ---
    segments = [{
        "origin": origin_code,
        "destination": dest_code,
        "departureDate": dep_date.isoformat()
    }]

    if ret_date:
        segments.append({
            "origin": dest_code,
            "destination": origin_code,
            "departureDate": ret_date.isoformat()
        })

    # --- Flight Query Helper ---
    def fetch_flights(seg):
        params = {
            "originLocationCode": seg["origin"],
            "destinationLocationCode": seg["destination"],
            "departureDate": seg["departureDate"],
            "adults": pax_adults,
            "children": pax_children,
            "infants": pax_infants,
            "travelClass": travel_class,
            "currencyCode": currency,
            "max": 10
        }
        if nonstop:
            params["nonStop"] = True

        try:
            raw = query_amadeus_flights(params)
            offers = raw.get("data", [])

            if not offers and nonstop:
                print("[RETRY] Nonstop empty → retry with stopovers")
                params.pop("nonStop", None)
                raw = query_amadeus_flights(params)
                offers = raw.get("data", [])

            return format_amadeus_response(raw, params) if offers else []

        except Exception as e:
            print("[ERROR] Query failed:", e)
            return []

    # --- Execute actual searches ---
    flight_results = []
    for seg in segments:
        offers = fetch_flights(seg)
        seg_copy = dict(seg)

        if not offers:
            seg_copy["offers"] = []
            seg_copy["no_flights"] = True
            seg_copy["message"] = "No flights found for this route."
        else:
            seg_copy["offers"] = offers
            seg_copy["no_flights"] = False

        flight_results.append(seg_copy)

    # --- Build message ---
    def fmt_date(d):
        return datetime.strptime(d, "%Y-%m-%d").strftime("%d %b %Y")

    if ret_date:
        msg = (
            f"Here are your {travel_class.title()} fares in {currency}: "
            f"{origin_code} → {dest_code} on {fmt_date(dep_date.isoformat())} "
            f"and back on {fmt_date(ret_date.isoformat())}."
        )
    else:
        if all(seg.get("no_flights") for seg in flight_results):
            msg = (
                f"⚠️ No flights found for {origin_code} → {dest_code} "
                f"on {fmt_date(dep_date.isoformat())}."
            )
        else:
            msg = (
                f"Here are your one-way {travel_class.title()} fares in {currency}: "
                f"{origin_code} → {dest_code} on {fmt_date(dep_date.isoformat())}."
            )

    # --- Final payload ---
    final_payload = {
        "segments": flight_results,
        "flights": flight_results
    }

    print("[FLIGHT SEARCH DONE]")
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

    # =====================================================
    # ⚠️ HOTELS
    # =====================================================
import time
import hashlib
import requests
from flask import jsonify, request



def hotelbeds_signature():
    ts = str(int(time.time()))
    raw = HOTELBEDS_API_KEY + HOTELBEDS_SECRET + ts
    return hashlib.sha256(raw.encode('utf-8')).hexdigest(), ts




@app.route("/api/hotels", methods=["POST"])
def api_search_hotels(data=None):
    from flask import request, jsonify

    print("\n[HOTELBEDS] Incoming hotel search request...")

    # Allow internal GPT router OR external POST request
    if data is None:
        data = request.get_json() or {}

    print("[HOTELBEDS] Received data:", data)

    # Extract city
    city = data.get("city")

    # =================================================================
    # 🧭 HOTEL DATE HANDLER — ISO + normalize_date + nights
    # =================================================================
    raw_check_in  = data.get("checkIn")
    raw_check_out = data.get("checkOut")
    nights = data.get("nights", 1)

    check_in = None
    check_out = None

    # 1️⃣ Try ISO format first
    def try_iso(d):
        try:
            return datetime.fromisoformat(str(d)).date()
        except:
            return None

    iso_in  = try_iso(raw_check_in)
    iso_out = try_iso(raw_check_out)

    if iso_in:
        check_in = iso_in.strftime("%Y-%m-%d")
    if iso_out:
        check_out = iso_out.strftime("%Y-%m-%d")

    # 2️⃣ If ISO missing → use flight parser
    if not iso_in or not iso_out:
        normalized = normalize_date(f"{raw_check_in} to {raw_check_out}")
        if normalized:
            check_in = normalized[0]
            if len(normalized) > 1:
                check_out = normalized[1]

    # 3️⃣ If only checkIn provided → compute nights
    if check_in and not check_out:
        d = datetime.fromisoformat(check_in)
        check_out = (d + timedelta(days=nights)).strftime("%Y-%m-%d")

    print(f"[HOTELBEDS] Nights requested = {nights}")
    print(f"[HOTELBEDS] Parsed dates → checkIn={check_in}, checkOut={check_out}")

        # =================================================================
    # 🏨 HOTEL FUTURE DATE CORRECTION — PRESERVE NIGHTS
    # =================================================================
    today = datetime.now().date()

    ci = datetime.fromisoformat(check_in).date()
    co = datetime.fromisoformat(check_out).date()

    stay_nights = (co - ci).days  # ALWAYS preserve nights

    # CASE 1: BOTH dates are in the past → shift entire range forward
    if ci < today and co < today:
        # try same calendar dates this year
        ci = ci.replace(year=today.year)
        co = ci + timedelta(days=stay_nights)

        # if still in past (e.g., February < November) → shift to next year
        if co < today:
            ci = ci.replace(year=today.year + 1)
            co = ci + timedelta(days=stay_nights)

    # CASE 2: Check-in is in past but check-out is future → shift both
    elif ci < today <= co:
        ci = today
        co = ci + timedelta(days=stay_nights)

    # 🚫 CASE 3: NEVER collapse multi-night stays
    check_in  = ci.strftime("%Y-%m-%d")
    check_out = co.strftime("%Y-%m-%d")

    print(f"[HOTELBEDS] Future-corrected dates → checkIn={check_in}, checkOut={check_out}")


    # =================================================================
    # VALIDATION
    # =================================================================
    if not city or not check_in or not check_out:
        print("[HOTELBEDS] ❌ Missing required search fields.")
        return jsonify({
            "type": "text",
            "text": "⚠️ Missing hotel search information."
        })

    adults = data.get("adults", 1)
    rooms  = data.get("rooms", 1)

    # =================================================================
    # HOTELBEDS DESTINATION CODES
    # =================================================================
    def get_hotelbeds_city_code(city_name):
        mapping = {
            "paris": "PAR",
            "new york": "NYC",
            "dubai": "DXB",
            "london": "LON",
            "mumbai": "BOM",
            "delhi": "DEL",
            "bangkok": "BKK",
            "abu dhabi": "AUH",
            "berlin": "BER",
            "madrid": "MAD"
        }
        return mapping.get(city_name.lower().strip())

    destination_code = get_hotelbeds_city_code(city)

    if not destination_code:
        print("[HOTELBEDS] ❌ Unsupported city:", city)
        return jsonify({
            "type": "text",
            "text": f"❌ Sorry, I don’t support hotel searches in '{city}'."
        })

    print(f"[HOTELBEDS] Using destination code: {destination_code}")

    # =================================================================
    # HOTELBEDS REQUEST BODY
    # =================================================================
    body = {
        "stay": {
            "checkIn":  check_in,
            "checkOut": check_out
        },
        "occupancies": [
            {"rooms": rooms, "adults": adults, "children": 0}
        ],
        "destination": {
            "code": destination_code,
            "type": "SIMPLE"
        }
    }

    # =================================================================
    # HOTELBEDS SIGNATURE + HEADERS
    # =================================================================
    sig, ts = hotelbeds_signature()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Api-key": HOTELBEDS_API_KEY,
        "X-Signature": sig,
        "X-Timestamp": ts
    }

    print("[HOTELBEDS] Sending request to API...")

    # =================================================================
    # HOTELBEDS API CALL
    # =================================================================
    try:
        resp = requests.post(HOTELBEDS_ENDPOINT, headers=headers, json=body)
        print("[HOTELBEDS] Response Status:", resp.status_code)
        resp.raise_for_status()

        response_json = resp.json()

        raw_hotels = response_json.get("hotels", {}).get("hotels", [])
        print(f"[HOTELBEDS] Found {len(raw_hotels)} hotels.")

        raw_hotels = raw_hotels[:5]  # Limit
        print(f"[HOTELBEDS] Returning {len(raw_hotels)} hotels after limit.")

    except Exception as e:
        print("[HOTELBEDS] ERROR:", e)
        return jsonify({
            "type": "text",
            "text": f"❌ Hotel search failed: {str(e)}"
        })

    # =================================================================
    # TRANSFORM HOTEL DATA FOR FRONTEND
    # =================================================================
    results = []

    for h in raw_hotels:
        hotel_code = h.get("code")
        images = get_hotelbeds_images(hotel_code)

        raw_addr = h.get("address", {})
        address = (
            raw_addr.get("content") or
            raw_addr.get("street") or
            raw_addr.get("fullAddress") or
            "Address unavailable"
        )

        facilities = h.get("facilities", [])
        amenity_list = [f.get("description") for f in facilities if "description" in f]

        rate_list = h.get("rates", [])
        price_block = rate_list[0] if rate_list else {}

        results.append({
            "id": hotel_code,
            "name": h.get("name"),
            "address": address,
            "rating": h.get("categoryName", "0").replace("stars", "").strip(),
            "latitude": h.get("coordinates", {}).get("latitude"),
            "longitude": h.get("coordinates", {}).get("longitude"),
            "images": images,
            "price": {
                "amount": extract_price(h),
                "currency": price_block.get("currency", "AED")
            },
            "nights": stay_nights,
            "description": h.get("description", {}).get("content", ""),
            "amenities": amenity_list,
            "bookingLink": price_block.get("rateKey", "")
        })

    print("[HOTELBEDS] ✔ Returning hotels with images.")

    return jsonify({
        "type": "hotels",
        "data": results
    })


def get_hotelbeds_images(hotel_code):
    if hotel_code in hotel_image_cache:
        return hotel_image_cache[hotel_code]

    try:
        sig, ts = hotelbeds_signature()

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Api-key": HOTELBEDS_API_KEY,
            "X-Signature": sig,
            "X-Timestamp": ts
        }

        url = f"https://api.test.hotelbeds.com/hotel-content-api/1.0/hotels/{hotel_code}/details"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        images = data.get("hotel", {}).get("images", [])
        #print("🔥 HOTELBEDS RAW IMAGES:", images)

        urls = []

        for img in images:
            path = img.get("path")
            if not path:
                continue

            # Normalize path → always force "/giata/"
            if not path.startswith("/"):
                path = "/" + path

            full_url = "https://photos.hotelbeds.com/giata" + path

            urls.append(full_url)

        if not urls:
            urls = ["/static/hotel-placeholder.jpg"]

        hotel_image_cache[hotel_code] = urls[:5]
        return urls[:5]

    except Exception as e:
        print("[HOTELBEDS IMG ERROR]", e)
        return ["/static/hotel-placeholder.jpg"]

def get_hotelbeds_room_images(hotel_code, room_code):
    try:
        sig, ts = hotelbeds_signature()

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Api-key": HOTELBEDS_API_KEY,
            "X-Signature": sig,
            "X-Timestamp": ts
        }

        url = f"https://api.test.hotelbeds.com/hotel-content-api/1.0/hotels/{hotel_code}/details"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        room_list = data.get("rooms", [])
        images = []

        for room in room_list:
            if room.get("code") == room_code:
                for img in room.get("images", []):
                    path = img.get("path")
                    if not path:
                        continue
                    if not path.startswith("/"):
                        path = "/" + path

                    full_url = "https://photos.hotelbeds.com/giata" + path
                    images.append(full_url)

        if not images:
            return ["https://source.unsplash.com/800x600/?hotel-room"]

        return images[:5]

    except Exception as e:
        print("[HOTELBEDS ROOM IMG ERROR]", e)
        return ["https://source.unsplash.com/800x600/?hotel-room"]




def extract_price(hotel):
    """
    Universal Hotelbeds price extractor (NET, SELLING, PRICE object, RATES object).
    Returns a string numeric price.
    """

    # 1) New PRICE object
    if "price" in hotel:
        price = hotel["price"]
        return (
            price.get("sellingRate")
            or price.get("net")
            or price.get("hotelSellingRate")
            or price.get("hotelNet")
        )

    # 2) Standard RATES array
    rates = hotel.get("rates", [])
    if rates:
        r = rates[0]

        return (
            r.get("sellingRate")
            or r.get("net")
            or r.get("hotelSellingRate")
            or r.get("hotelNet")
            or r.get("amount")
        )

    # 3) Legacy structure (rare)
    if "minRate" in hotel:
        return hotel["minRate"]

    if "maxRate" in hotel:
        return hotel["maxRate"]

    # 4) Fallback
    return "0"


@app.route("/api/hotel/rooms", methods=["POST"])
def get_hotel_rooms():
    data = request.json
    hotel_id = data.get("hotel_id")   # Must match HotelBeds hotel code

    try:
        sig, ts = hotelbeds_signature()

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Api-key": HOTELBEDS_API_KEY,
            "X-Signature": sig,
            "X-Timestamp": ts
        }

        # Hotel details include ROOMS + IMAGES
        url = f"https://api.test.hotelbeds.com/hotel-content-api/1.0/hotels/{hotel_id}/details"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        rooms_json = data.get("rooms", [])
        rooms = []

        for r in rooms_json:
            room_code = r.get("code")

            # Fetch room images
            image_list = get_hotelbeds_room_images(hotel_id, room_code)

            rooms.append({
                "name": r.get("name", "Room"),
                "beds": r.get("characteristics", {}).get("beds", "Beds not provided"),
                "free_cancellation": True,  # If you want, we can map HotelBeds policies later.
                "price": "At hotel",        # Real price requires API from HotelBeds Booking API.
                "image": image_list[0] if image_list else None
            })

        return jsonify({"rooms": rooms})

    except Exception as e:
        print("[ROOM ENDPOINT ERROR]", e)
        return jsonify({"rooms": []})








@app.route("/api/transfers", methods=["POST"])
def search_transfers():
    data = request.json
    origin = data.get("origin")      # Airport IATA
    destination = data.get("destination")  # coordinates or city center
    date = data.get("date")
    passengers = data.get("passengers", 1)

    try:
        response = AMADEUS.shopping.transfer_offers.get(
            startLocationCode=origin,
            endLocationCode=destination,
            transferDate=date,
            passengers=passengers
        )
        return jsonify(response.data)

    except Exception as e:
        print("TRANSFER ERROR:", e)
        return jsonify({"error": "Transfer search failed"}), 500
# ===============================================================
# 🔹 Run Server
# ===============================================================
from flask import send_from_directory

@app.route("/")
def serve_index():
    """Serve the frontend index.html file when visiting the root URL."""
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )






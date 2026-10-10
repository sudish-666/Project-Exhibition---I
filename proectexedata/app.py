"""
Hyperlocal AQI prototype - DSN2098 Team 109

Loads the AQI data for the selected city, trains a Prophet model, and shows
the AQI + health advice for whichever week you pick. Advice changes based on
the profile.

To run it:
    python prep_data.py        (once, makes the per-city csv files)
    pip install streamlit prophet pandas matplotlib
    streamlit run app.py
(the aqi_<city>.csv files need to be in the same folder as this file)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# ---------------- settings ----------------
# city name shown in the dropdown -> csv file made by prep_data.py
CITY_FILES = {
    "Delhi": "aqi_delhi.csv",
    "Bengaluru": "aqi_bengaluru.csv",
}
DATE_COL = "date"
AQI_COL = "AQI"
MAX_FUTURE_DAYS = 90   # how many days past the end of the data you're allowed to pick


# ---------------- risk categories ----------------
# thresholds are the ones we used in the report (section 11)
def aqi_to_category(aqi):
    if aqi <= 100:
        return "Good"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    else:
        return "Severe"


# advice for each profile, for each category
ADVICE = {
    "General Public": {
        "Good": "Air quality is satisfactory. No precautions needed.",
        "Moderate": "Acceptable air quality. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
        "Poor": "Reduce prolonged or heavy outdoor exertion. Consider wearing a mask if outside for long periods.",
        "Severe": "Avoid outdoor activity where possible. Keep windows closed and use an air purifier if available.",
    },
    "Asthma Patients": {
        "Good": "Safe to go about your normal routine, including outdoor exercise.",
        "Moderate": "Keep your reliever inhaler on hand. Avoid heavy outdoor exertion during peak traffic hours.",
        "Poor": "Avoid outdoor activity. Stay indoors with windows closed. Keep rescue medication accessible at all times.",
        "Severe": "Do not go outside unless necessary. Wear an N95 mask if you must step out. Contact your doctor if symptoms appear.",
    },
    "Outdoor Workers": {
        "Good": "Normal working conditions. No special precautions needed.",
        "Moderate": "Take regular breaks in shaded or indoor areas if work is strenuous.",
        "Poor": "Wear a mask during work. Take frequent breaks indoors. Reschedule non-essential outdoor tasks if possible.",
        "Severe": "Employers should consider rescheduling outdoor work. If unavoidable, use N95 masks and limit continuous exposure to under 1 hour at a time.",
    },
    "Students": {
        "Good": "Normal outdoor activities and sports are safe.",
        "Moderate": "Outdoor sports are fine, but avoid very strenuous activity for extended periods.",
        "Poor": "Move outdoor sports/PE classes indoors where possible. Avoid long periods of outdoor play.",
        "Severe": "Keep outdoor recess and sports indoors. Schools should consider informing parents of the day's risk level.",
    },
}

USER_GROUPS = list(ADVICE.keys())


# ---------------- data + model ----------------
# takes the file path as an argument, so each city gets its own cached copy
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])
    df = df[[DATE_COL, AQI_COL]].dropna()   # drop days with no reading
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


# cached so it only trains once per city, not every time the date changes
@st.cache_resource
def train_model(df, city):   # city isn't used inside, it just keeps the cache entries separate
    # prophet wants the columns named ds and y
    prophet_df = df.rename(columns={DATE_COL: "ds", AQI_COL: "y"})
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    return model


def get_week_data(model, df, week_start):
    """Gets 7 days from week_start: the model's prediction, plus the real AQI if we have it."""
    dates = pd.date_range(week_start, periods=7)
    pred = model.predict(pd.DataFrame({"ds": dates}))[["ds", "yhat"]]
    pred = pred.rename(columns={"ds": "date", "yhat": "predicted_AQI"})

    # left merge, so days with no real data just end up blank
    actual = df.rename(columns={DATE_COL: "date", AQI_COL: "actual_AQI"})
    out = pred.merge(actual, on="date", how="left")

    # if the real value exists use it, otherwise fall back to the prediction
    out["AQI_used"] = out["actual_AQI"].fillna(out["predicted_AQI"])
    out["Source"] = out["actual_AQI"].notna().map({True: "Actual", False: "Forecast"})
    out["risk_category"] = out["AQI_used"].apply(aqi_to_category)
    return out


# ---------------- dashboard ----------------
st.set_page_config(page_title="Hyperlocal AQI Forecast", layout="centered")
st.title("Hyperlocal Air Quality & Health Risk Forecast")
st.caption("DSN2098 — Team 109 — Prototype")

# city goes first because everything below depends on it
selected_city = st.selectbox("Select city:", list(CITY_FILES.keys()))
csv_path = CITY_FILES[selected_city]

try:
    df = load_data(csv_path)
except FileNotFoundError:
    st.error(f"Could not find '{csv_path}'. Run prep_data.py first so the city files get created.")
    st.stop()

st.write(f"Loaded {len(df)} days of historical AQI data for {selected_city}, from {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}.")

with st.spinner(f"Training forecasting model for {selected_city}..."):
    model = train_model(df, selected_city)

# --- profile + week picker ---
min_date = df[DATE_COL].min().date()
max_date = df[DATE_COL].max().date()
latest_allowed = (df[DATE_COL].max() + pd.Timedelta(days=MAX_FUTURE_DAYS)).date()

# start on the last 7 days of data, but don't go before the first date
default_start = max(min_date, (df[DATE_COL].max() - pd.Timedelta(days=6)).date())

col1, col2 = st.columns(2)
with col1:
    selected_group = st.selectbox("Select your profile:", USER_GROUPS)
with col2:
    week_start = st.date_input(
        "Select week start date:",
        value=default_start,
        min_value=min_date,
        max_value=latest_allowed,
        key=f"week_start_{selected_city}",   # new key per city so the date resets when you switch
    )

week_start = pd.Timestamp(week_start)
week_end = week_start + pd.Timedelta(days=6)
st.caption(f"Showing: {week_start.strftime('%d %b %Y')} to {week_end.strftime('%d %b %Y')}")

week_df = get_week_data(model, df, week_start)

# --- quick summary for the week ---
avg_aqi = week_df["AQI_used"].mean()
worst = week_df.loc[week_df["AQI_used"].idxmax()]
c1, c2 = st.columns(2)
# delta_color="off" so streamlit doesn't colour the label green/red like it's a number change
c1.metric("Average AQI for the week", f"{avg_aqi:.0f}", aqi_to_category(avg_aqi), delta_color="off")
c2.metric("Worst day", worst["date"].strftime("%d %b"), f"AQI {worst['AQI_used']:.0f}", delta_color="off")

# advice is based on the worst day, so we don't undersell a bad day
st.info(ADVICE[selected_group][aqi_to_category(worst["AQI_used"])])

# --- graph: the 30 days before the week, then the week itself ---
st.subheader(f"AQI Trend for Selected Week - {selected_city}")
fig, ax = plt.subplots(figsize=(8, 4))

before = df[(df[DATE_COL] >= week_start - pd.Timedelta(days=30)) & (df[DATE_COL] < week_start)]
if not before.empty:
    ax.plot(before[DATE_COL], before[AQI_COL], label="Previous 30 days (actual)", color="#1F5C4C")

ax.plot(week_df["date"], week_df["predicted_AQI"], label="Model prediction", color="#E67E22", linestyle="--", marker="o")

# only draw the actual line if the selected week has real data
actual_part = week_df.dropna(subset=["actual_AQI"])
if not actual_part.empty:
    ax.plot(actual_part["date"], actual_part["actual_AQI"], label="Actual (selected week)", color="#2980B9", marker="s")

# category boundary lines
for y in (100, 200, 300):
    ax.axhline(y, color="gray", linestyle=":", linewidth=1)

ax.set_ylabel("AQI")
ax.legend()
fig.autofmt_xdate()   # tilts the dates so they stop overlapping
st.pyplot(fig)

# --- day by day table ---
st.subheader(f"Day-by-Day Breakdown for {selected_group} - {selected_city}")
display_df = week_df.copy()
display_df["Advice"] = display_df["risk_category"].apply(lambda c: ADVICE[selected_group][c])
display_df["Date"] = display_df["date"].dt.strftime("%a, %d %b %Y")
display_df["Predicted AQI"] = display_df["predicted_AQI"].round(0)
display_df["Actual AQI"] = display_df["actual_AQI"].round(0)
display_df = display_df.rename(columns={"risk_category": "Risk Category"})
st.dataframe(
    display_df[["Date", "Source", "Predicted AQI", "Actual AQI", "Risk Category", "Advice"]],
    hide_index=True, use_container_width=True
)
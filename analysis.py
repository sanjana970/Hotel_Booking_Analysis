import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10,6)


# Load dataset

df = pd.read_csv("Hotel_bookings_final.csv")


# Data Preparation

df['is_canceled'] = df['booking_status'].apply(lambda x: 1 if x == "Cancelled" else 0)

# Check if arrival_date exists
if 'arrival_date' in df.columns:
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df['arrival_month'] = df['arrival_date'].dt.month
    has_arrival_date = True
else:
    print("⚠️ 'arrival_date' column not found. Skipping monthly/seasonal analysis.")
    has_arrival_date = False


# Basic Info

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nInfo:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())


# Key Metrics

cancel_rate = df['is_canceled'].mean() * 100
print(f"\nOverall Cancellation Rate: {cancel_rate:.2f}%")


# 1. Booking Distribution

# a) By Channel
plt.figure()
sns.countplot(data=df, x="booking_channel", order=df['booking_channel'].value_counts().index, palette="viridis")
plt.title("Bookings by Channel", fontsize=16)
plt.xlabel("Booking Channel")
plt.ylabel("Number of Bookings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bookings_by_channel.png")
plt.close()

# b) By Room Type
plt.figure()
sns.countplot(data=df, x="room_type", order=df['room_type'].value_counts().index, palette="magma")
plt.title("Bookings by Room Type", fontsize=16)
plt.xlabel("Room Type")
plt.ylabel("Number of Bookings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bookings_by_room.png")
plt.close()

# c) By Star Rating
if 'star_rating' in df.columns:
    plt.figure()
    sns.countplot(data=df, x="star_rating", palette="coolwarm", order=sorted(df['star_rating'].unique()))
    plt.title("Bookings by Hotel Star Rating", fontsize=16)
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Bookings")
    plt.tight_layout()
    plt.savefig("bookings_by_star.png")
    plt.close()


# 2. Cancellation Analysis

# a) Overall
plt.figure()
sns.countplot(data=df, x="is_canceled", palette="Set2")
plt.title("Overall Cancellations (0=No, 1=Yes)", fontsize=16)
plt.xlabel("Canceled")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.savefig("cancellations_overall.png")
plt.close()

# b) By Booking Channel
plt.figure()
sns.barplot(data=df, x="booking_channel", y="is_canceled", palette="viridis", ci=None)
plt.title("Cancellation Rate by Booking Channel", fontsize=16)
plt.xlabel("Booking Channel")
plt.ylabel("Cancellation Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("cancellation_by_channel.png")
plt.close()

# c) By Room Type
plt.figure()
sns.barplot(data=df, x="room_type", y="is_canceled", palette="magma", ci=None)
plt.title("Cancellation Rate by Room Type", fontsize=16)
plt.xlabel("Room Type")
plt.ylabel("Cancellation Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("cancellation_by_room.png")
plt.close()

# d) By Star Rating
if 'star_rating' in df.columns:
    plt.figure()
    sns.barplot(data=df, x="star_rating", y="is_canceled", palette="coolwarm", ci=None)
    plt.title("Cancellation Rate by Hotel Star Rating", fontsize=16)
    plt.xlabel("Star Rating")
    plt.ylabel("Cancellation Rate")
    plt.tight_layout()
    plt.savefig("cancellation_by_star.png")
    plt.close()

# e) Lead Time vs Cancellations
if 'lead_time' in df.columns:
    plt.figure()
    sns.histplot(df[df['is_canceled']==1]['lead_time'], bins=50, kde=True, color='tomato')
    plt.title("Lead Time Distribution for Cancellations", fontsize=16)
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Number of Cancellations")
    plt.tight_layout()
    plt.savefig("leadtime_cancellations.png")
    plt.close()

# f) ADR Distribution
if 'average_daily_rate' in df.columns:
    plt.figure()
    sns.histplot(df['average_daily_rate'], bins=50, kde=True, color='teal')
    plt.title("Average Daily Rate Distribution", fontsize=16)
    plt.xlabel("Average Daily Rate")
    plt.ylabel("Number of Bookings")
    plt.tight_layout()
    plt.savefig("adr_distribution.png")
    plt.close()

# g) ADR vs Star Rating & Cancellation
if 'average_daily_rate' in df.columns and 'star_rating' in df.columns:
    plt.figure()
    sns.boxplot(data=df, x="star_rating", y="average_daily_rate", hue="is_canceled", palette="Set3")
    plt.title("ADR by Star Rating and Cancellation", fontsize=16)
    plt.xlabel("Star Rating")
    plt.ylabel("Average Daily Rate")
    plt.tight_layout()
    plt.savefig("adr_by_star_cancellation.png")
    plt.close()

# 3. Temporal / Seasonal Trends

if has_arrival_date:
    monthly_cancel = df.groupby('arrival_month')['is_canceled'].mean()
    plt.figure()
    monthly_cancel.plot(kind='bar', color='skyblue')
    plt.title("Monthly Cancellation Rate", fontsize=16)
    plt.xlabel("Month")
    plt.ylabel("Cancellation Rate")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("monthly_cancellation_rate.png")
    plt.close()

# --------------------
# 4. Correlation Heatmap
# --------------------
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

print("Analysis complete.")

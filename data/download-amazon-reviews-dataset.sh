#!/usr/bin/env bash
# Amazon Reviews 2023 — bulk downloader for reviews + metadata per category
# Source of links/categories: https://amazon-reviews-2023.github.io/
# Output:
#   data/reviews/<Category>.jsonl.gz
#   data/meta/meta_<Category>.jsonl.gz

set -euo pipefail

BASE="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
REV_BASE="${BASE}/review_categories"
META_BASE="${BASE}/meta_categories"

# Create output dirs
mkdir -p data/reviews data/meta

# All categories listed on the site (33 categories + "Unknown")
categories=(
  All_Beauty
  Amazon_Fashion
  Appliances
  Arts_Crafts_and_Sewing
  Automotive
  Baby_Products
  Beauty_and_Personal_Care
  Books
  CDs_and_Vinyl
  Cell_Phones_and_Accessories
  Clothing_Shoes_and_Jewelry
  Digital_Music
  Electronics
  Gift_Cards
  Grocery_and_Gourmet_Food
  Handmade_Products
  Health_and_Household
  Health_and_Personal_Care
  Home_and_Kitchen
  Industrial_and_Scientific
  Kindle_Store
  Magazine_Subscriptions
  Movies_and_TV
  Musical_Instruments
  Office_Products
  Patio_Lawn_and_Garden
  Pet_Supplies
  Software
  Sports_and_Outdoors
  Subscription_Boxes
  Tools_and_Home_Improvement
  Toys_and_Games
  Video_Games
  Unknown
)

# Download with resume, a few retries, and a sane timeout
for c in "${categories[@]}"; do
  review_url="${REV_BASE}/${c}.jsonl.gz"
  meta_url="${META_BASE}/meta_${c}.jsonl.gz"

  echo "==> ${c} (reviews)"
  wget -c --tries=3 --timeout=30 -P data/reviews "${review_url}"

  echo "==> ${c} (metadata)"
  wget -c --tries=3 --timeout=30 -P data/meta "${meta_url}"
done

echo "All done. Files are in ./data/reviews and ./data/meta"

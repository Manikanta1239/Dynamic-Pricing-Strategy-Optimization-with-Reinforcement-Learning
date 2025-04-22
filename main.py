"""
Dynamic Pricing System with Reinforcement Learning through Web Scraping
-----------------------------------------------------------------------
This system combines web scraping, data processing, and reinforcement learning
to create an intelligent dynamic pricing strategy based on market conditions.
"""

import os
import time
import random
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# Web scraping dependencies
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent

# ML and RL dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#------------------------------------------------------------------------------
# Web Scraping Components
#------------------------------------------------------------------------------

class ProxyManager:
    """Manages a pool of proxy servers for web scraping."""
    
    def __init__(self, proxy_list_file=None):
        """Initialize with a file containing proxy servers or use defaults."""
        self.proxies = []
        self.current_index = 0
        
        if proxy_list_file and os.path.exists(proxy_list_file):
            with open(proxy_list_file, 'r') as f:
                self.proxies = [line.strip() for line in f if line.strip()]
        else:
            # Default free proxies (these would need to be updated regularly)
            self.proxies = [
                "http://203.24.108.172:80",
                "http://45.79.158.95:44554",
                # Add more default proxies as needed
            ]
        
        self.proxy_count = len(self.proxies)
        print(f"Initialized proxy manager with {self.proxy_count} proxies")
    
    def get_proxy(self):
        """Get the next proxy from the rotation."""
        if not self.proxies:
            return None
        
        proxy = self.proxies[self.current_index]
        self.current_index = (self.current_index + 1) % self.proxy_count
        return {"http": proxy, "https": proxy}
    
    def mark_proxy_failed(self, proxy):
        """Mark a proxy as failed and remove it from the rotation."""
        if proxy["http"] in self.proxies:
            self.proxies.remove(proxy["http"])
            self.proxy_count = len(self.proxies)
            print(f"Removed failed proxy. {self.proxy_count} proxies remaining.")


class UserAgentRotator:
    """Provides rotating user agents for web scraping."""
    
    def __init__(self):
        """Initialize with a list of common user agents."""
        self.ua = UserAgent()
    
    def get_random_user_agent(self):
        """Get a random user agent."""
        return self.ua.random


class ScraperBase:
    """Base class for website scrapers with common functionality."""
    
    def __init__(self, proxy_manager=None, use_selenium=False):
        """Initialize the base scraper with proxy support and rate limiting."""
        self.proxy_manager = proxy_manager or ProxyManager()
        self.ua_rotator = UserAgentRotator()
        self.use_selenium = use_selenium
        self.driver = None
        self.session = requests.Session()
        
        # Rate limiting parameters
        self.min_delay = 2  # Minimum delay between requests in seconds
        self.max_delay = 5  # Maximum delay between requests in seconds
        self.last_request_time = 0
        
        # Initialize Selenium if needed
        if self.use_selenium:
            self._setup_selenium()
    
    def _setup_selenium(self):
        """Set up the Selenium WebDriver."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"user-agent={self.ua_rotator.get_random_user_agent()}")
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
    
    def _wait_between_requests(self):
        """Implement rate limiting by waiting between requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_delay:
            wait_time = random.uniform(self.min_delay, self.max_delay)
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def get_page(self, url):
        """Get a page using either requests or Selenium with proxy support."""
        self._wait_between_requests()
        
        headers = {
            "User-Agent": self.ua_rotator.get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "DNT": "1",  # Do Not Track
        }
        
        if self.use_selenium:
            try:
                self.driver.get(url)
                # Wait for page to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                return self.driver.page_source
            except Exception as e:
                print(f"Selenium error: {e}")
                return None
        else:
            for _ in range(3):  # Retry up to 3 times
                proxy = self.proxy_manager.get_proxy()
                try:
                    response = self.session.get(
                        url, 
                        headers=headers,
                        proxies=proxy,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        return response.text
                    else:
                        print(f"Request failed with status {response.status_code}")
                        time.sleep(random.uniform(1, 3))
                except Exception as e:
                    print(f"Request error: {e}")
                    if proxy:
                        self.proxy_manager.mark_proxy_failed(proxy)
                    time.sleep(random.uniform(1, 3))
            
            return None
    
    def parse_page(self, html_content):
        """Parse the HTML content to extract relevant data."""
        raise NotImplementedError("Subclasses must implement parse_page method")
    
    def scrape_product(self, product_url):
        """Scrape data for a single product."""
        html_content = self.get_page(product_url)
        if html_content:
            return self.parse_page(html_content)
        return None
    
    def close(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
        self.session.close()


class AmazonScraper(ScraperBase):
    """Scraper specifically for Amazon product pages."""
    
    def __init__(self, proxy_manager=None):
        """Initialize with Amazon-specific settings."""
        super().__init__(proxy_manager, use_selenium=True)
        
        # Amazon-specific selectors
        self.selectors = {
            "product_title": "#productTitle",
            "price": "#priceblock_ourprice, .a-price .a-offscreen",
            "rating": "span.a-icon-alt",
            "review_count": "#acrCustomerReviewText",
            "availability": "#availability",
            "features": "#feature-bullets .a-list-item",
            "category": "#wayfinding-breadcrumbs_feature_div ul li:not(:last-child) a",
        }
    
    def parse_page(self, html_content):
        """Parse Amazon product page HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract product data
        result = {
            "timestamp": datetime.now().isoformat(),
            "source": "amazon",
            "url": self.driver.current_url if self.driver else "",
        }
        
        # Extract product title
        title_element = soup.select_one(self.selectors["product_title"])
        result["title"] = title_element.get_text().strip() if title_element else None
        
        # Extract price
        price_element = soup.select_one(self.selectors["price"])
        if price_element:
            price_text = price_element.get_text().strip()
            # Remove currency symbols and convert to float
            price_value = ''.join(c for c in price_text if c.isdigit() or c == '.')
            try:
                result["price"] = float(price_value)
            except ValueError:
                result["price"] = None
        else:
            result["price"] = None
        
        # Extract rating
        rating_element = soup.select_one(self.selectors["rating"])
        if rating_element:
            rating_text = rating_element.get_text().strip()
            try:
                result["rating"] = float(rating_text.split(' ')[0])
            except (ValueError, IndexError):
                result["rating"] = None
        else:
            result["rating"] = None
        
        # Extract review count
        review_count_element = soup.select_one(self.selectors["review_count"])
        if review_count_element:
            review_text = review_count_element.get_text().strip()
            try:
                result["review_count"] = int(''.join(c for c in review_text if c.isdigit()))
            except ValueError:
                result["review_count"] = None
        else:
            result["review_count"] = None
        
        # Extract availability
        availability_element = soup.select_one(self.selectors["availability"])
        result["availability"] = availability_element.get_text().strip() if availability_element else None
        
        # Extract features
        feature_elements = soup.select(self.selectors["features"])
        result["features"] = [element.get_text().strip() for element in feature_elements]
        
        # Extract category
        category_elements = soup.select(self.selectors["category"])
        result["categories"] = [element.get_text().strip() for element in category_elements]
        
        return result


class FlipkartScraper(ScraperBase):
    """Scraper specifically for Flipkart product pages."""
    
    def __init__(self, proxy_manager=None):
        """Initialize with Flipkart-specific settings."""
        super().__init__(proxy_manager, use_selenium=True)
        
        # Flipkart-specific selectors
        self.selectors = {
            "product_title": "span.B_NuCI",
            "price": "div._30jeq3._16Jk6d",
            "original_price": "div._3I9_wc._2p6lqe",
            "rating": "div._3LWZlK",
            "review_count": "span._2_R_DZ",
            "availability": "div._16FRp0",
            "features": "div._2418kt li",
            "category": "div._1MR4o5 a",
        }
    
    def parse_page(self, html_content):
        """Parse Flipkart product page HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract product data
        result = {
            "timestamp": datetime.now().isoformat(),
            "source": "flipkart",
            "url": self.driver.current_url if self.driver else "",
        }
        
        # Extract product title
        title_element = soup.select_one(self.selectors["product_title"])
        result["title"] = title_element.get_text().strip() if title_element else None
        
        # Extract price
        price_element = soup.select_one(self.selectors["price"])
        if price_element:
            price_text = price_element.get_text().strip()
            # Remove currency symbols and convert to float
            price_value = ''.join(c for c in price_text if c.isdigit() or c == '.')
            try:
                result["price"] = float(price_value)
            except ValueError:
                result["price"] = None
        else:
            result["price"] = None
        
        # Extract original price (for discount calculation)
        original_price_element = soup.select_one(self.selectors["original_price"])
        if original_price_element:
            price_text = original_price_element.get_text().strip()
            price_value = ''.join(c for c in price_text if c.isdigit() or c == '.')
            try:
                result["original_price"] = float(price_value)
                # Calculate discount percentage
                if result["price"] and result["original_price"]:
                    result["discount"] = round(
                        (result["original_price"] - result["price"]) / result["original_price"] * 100, 2
                    )
            except ValueError:
                result["original_price"] = None
                result["discount"] = None
        else:
            result["original_price"] = None
            result["discount"] = None
        
        # Extract rating
        rating_element = soup.select_one(self.selectors["rating"])
        if rating_element:
            try:
                result["rating"] = float(rating_element.get_text().strip())
            except ValueError:
                result["rating"] = None
        else:
            result["rating"] = None
        
        # Extract review count
        review_count_element = soup.select_one(self.selectors["review_count"])
        if review_count_element:
            review_text = review_count_element.get_text().strip()
            try:
                # Extract just the numbers from text like "1,234 Reviews & 5,678 Ratings"
                numbers = ''.join(c if c.isdigit() else ' ' for c in review_text).split()
                if numbers:
                    result["review_count"] = int(numbers[0])
            except (ValueError, IndexError):
                result["review_count"] = None
        else:
            result["review_count"] = None
        
        # Extract availability
        availability_element = soup.select_one(self.selectors["availability"])
        result["availability"] = availability_element.get_text().strip() if availability_element else "In Stock"
        
        # Extract features
        feature_elements = soup.select(self.selectors["features"])
        result["features"] = [element.get_text().strip() for element in feature_elements]
        
        # Extract category
        category_elements = soup.select(self.selectors["category"])
        result["categories"] = [element.get_text().strip() for element in category_elements]
        
        return result


class ScraperManager:
    """Manages multiple scrapers and aggregates results."""
    
    def __init__(self):
        """Initialize with different platform scrapers."""
        proxy_manager = ProxyManager()
        self.scrapers = {
            "amazon": AmazonScraper(proxy_manager),
            "flipkart": FlipkartScraper(proxy_manager)
        }
        self.data_file = "product_data.csv"
        
        # Create data file if it doesn't exist
        if not os.path.exists(self.data_file):
            pd.DataFrame(columns=[
                "timestamp", "source", "url", "title", "price", "original_price",
                "discount", "rating", "review_count", "availability", "features", "categories"
            ]).to_csv(self.data_file, index=False)
    
    def scrape_product_across_platforms(self, product_urls):
        """Scrape the same product from different platforms."""
        results = []
        
        for platform, url in product_urls.items():
            if platform in self.scrapers:
                print(f"Scraping {platform}: {url}")
                result = self.scrapers[platform].scrape_product(url)
                if result:
                    results.append(result)
        
        # Save results to CSV
        if results:
            df = pd.DataFrame(results)
            existing_df = pd.read_csv(self.data_file)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_csv(self.data_file, index=False)
        
        return results
    
    def close(self):
        """Close all scrapers."""
        for scraper in self.scrapers.values():
            scraper.close()


#------------------------------------------------------------------------------
# Data Processing Components
#------------------------------------------------------------------------------

class DataProcessor:
    """Processes and analyzes scraped e-commerce data."""
    
    def __init__(self, data_file="product_data.csv"):
        """Initialize with the data file path."""
        self.data_file = data_file
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load data from CSV file."""
        if os.path.exists(self.data_file):
            return pd.read_csv(self.data_file)
        return None
    
    def clean_data(self, df):
        """Clean and preprocess the data."""
        if df is None or df.empty:
            return None
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Handle missing values
        df['price'] = df['price'].fillna(df['price'].median())
        df['rating'] = df['rating'].fillna(df['rating'].median())
        df['review_count'] = df['review_count'].fillna(df['review_count'].median())
        
        # Convert features and categories from string to list if needed
        for col in ['features', 'categories']:
            if col in df.columns and isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        # Create new features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # For products with multiple records, calculate price volatility
        if len(df) > 1:
            price_stats = df.groupby(['title', 'source'])['price'].agg(['mean', 'std', 'min', 'max'])
            price_stats.columns = ['price_mean', 'price_std', 'price_min', 'price_max']
            df = df.merge(price_stats, left_on=['title', 'source'], right_index=True)
            df['price_volatility'] = df['price_std'] / df['price_mean']
        
        return df
    
    def extract_features(self, df):
        """Extract features for the RL model."""
        if df is None or df.empty:
            return None, None
        
        # Select relevant features
        features = df[['price', 'rating', 'review_count', 'day_of_week', 'month', 'is_weekend']]
        
        # Add one-hot encoding for categorical features
        if 'source' in df.columns:
            source_dummies = pd.get_dummies(df['source'], prefix='source')
            features = pd.concat([features, source_dummies], axis=1)
        
        # Normalize numerical features
        numerical_cols = ['price', 'rating', 'review_count']
        features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
        
        return features, df['price'].values
    
    def split_train_test(self, features, prices, test_size=0.2):
        """Split data into training and testing sets."""
        return train_test_split(features, prices, test_size=test_size, random_state=42)
    
    def get_competitor_prices(self, df, product_title):
        """Get competitor prices for a specific product."""
        if df is None or df.empty:
            return {}
        
        # Filter for the specific product
        product_df = df[df['title'] == product_title]
        
        # Group by source and get latest price
        latest_prices = {}
        for source, group in product_df.groupby('source'):
            latest = group.loc[group['timestamp'].idxmax()]
            latest_prices[source] = latest['price']
        
        return latest_prices
    
    def analyze_price_elasticity(self, df):
        """Analyze price elasticity based on historical data."""
        if df is None or df.empty or len(df) < 10:
            return None
        
        # This is a simplified approach - in a real system you would use sales data
        # Here we'll use review_count as a proxy for popularity/demand
        
        # Group by price ranges and calculate average review_count
        df['price_bin'] = pd.qcut(df['price'], 5)
        elasticity_data = df.groupby('price_bin')['review_count'].mean().reset_index()
        
        # Calculate mid-point of each bin
        elasticity_data['price_mid'] = elasticity_data['price_bin'].apply(
            lambda x: (x.left + x.right) / 2
        )
        
        # Calculate price elasticity (percentage change in demand / percentage change in price)
        # This is a very simplified calculation
        elasticity_data = elasticity_data.sort_values('price_mid')
        elasticity_data['price_pct_change'] = elasticity_data['price_mid'].pct_change()
        elasticity_data['demand_pct_change'] = elasticity_data['review_count'].pct_change()
        elasticity_data['elasticity'] = elasticity_data['demand_pct_change'] / elasticity_data['price_pct_change']
        
        return elasticity_data
    
    def get_product_statistics(self):
        """Calculate statistics for each product."""
        df = self.load_data()
        if df is None or df.empty:
            return {}
        
        df = self.clean_data(df)
        
        # Group by product title
        stats = {}
        for title, group in df.groupby('title'):
            product_stats = {
                'avg_price': group['price'].mean(),
                'min_price': group['price'].min(),
                'max_price': group['price'].max(),
                'price_volatility': group['price'].std() / group['price'].mean() if group['price'].mean() > 0 else 0,
                'avg_rating': group['rating'].mean(),
                'total_reviews': group['review_count'].sum(),
                'sources': group['source'].unique().tolist(),
            }
            stats[title] = product_stats
        
        return stats


#------------------------------------------------------------------------------
# Reinforcement Learning Components
#------------------------------------------------------------------------------

class PricingEnvironment(gym.Env):
    """Custom OpenAI Gym environment for pricing optimization."""
    
    def __init__(self, 
                 product_data, 
                 price_range=(0.5, 2.0),
                 episode_length=30,
                 price_elasticity=-1.2):
        """
        Initialize the pricing environment.
        
        Args:
            product_data: DataFrame with product information
            price_range: Tuple of (min_factor, max_factor) relative to base price
            episode_length: Number of steps per episode
            price_elasticity: Default price elasticity if no data available
        """
        super(PricingEnvironment, self).__init__()
        
        self.product_data = product_data
        self.data_processor = DataProcessor()
        self.price_range = price_range
        self.episode_length = episode_length
        self.default_elasticity = price_elasticity
        
        # Calculate base price as average of observed prices
        self.base_price = self.product_data['price'].mean()
        
        # Define action and observation spaces
        # Action: Set price factor (between price_range[0] and price_range[1])
        self.action_space = spaces.Box(
            low=np.array([self.price_range[0]]),
            high=np.array([self.price_range[1]]),
            dtype=np.float32
        )
        
        # Observation: [current_price_factor, competitor_price_factor, day_of_week, month, is_weekend]
        self.observation_space = spaces.Box(
            low=np.array([self.price_range[0], self.price_range[0], 0, 1, 0]),
            high=np.array([self.price_range[1], self.price_range[1], 6, 12, 1]),
            dtype=np.float32
        )
        
        # Try to calculate price elasticity from data
        elasticity_data = self.data_processor.analyze_price_elasticity(self.product_data)
        if elasticity_data is not None and not elasticity_data['elasticity'].isna().all():
            # Use average elasticity from data
            valid_elasticity = elasticity_data['elasticity'].dropna()
            if len(valid_elasticity) > 0:
                self.price_elasticity = valid_elasticity.mean()
            else:
                self.price_elasticity = self.default_elasticity
        else:
            # Use default elasticity
            self.price_elasticity = self.default_elasticity
        
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        # Start with price factor of 1.0 (base price)
        self.current_price_factor = 1.0
        self.current_step = 0
        
        # Set up competitor price (simulated)
        self.competitor_price_factor = np.random.uniform(
            self.price_range[0], self.price_range[1]
        )
        
        # Set up calendar features
        self.current_date = datetime.now()
        self.day_of_week = self.current_date.weekday()
        self.month = self.current_date.month
        self.is_weekend = 1 if self.day_of_week >= 5 else 0
        
        # Create initial observation
        observation = np.array([
            self.current_price_factor,
            self.competitor_price_factor,
            self.day_of_week,
            self.month,
            self.is_weekend
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_demand(self, price_factor):
        """Calculate demand based on price elasticity model."""
        # Simple price elasticity model: Q2 = Q1 * (P2/P1)^elasticity
        
        # Base demand at price factor 1.0
        base_demand = 100
        
        # Calculate new demand using price elasticity
        demand = base_demand * (price_factor ** self.price_elasticity)
        
        # Add competitor price effect (cross-price elasticity)
        cross_elasticity = 0.8  # Positive because competitor's price increase helps our demand
        competitor_effect = (self.competitor_price_factor / price_factor) ** cross_elasticity
        demand *= competitor_effect
        
        # Add seasonality effects
        if self.is_weekend:
            demand *= 1.2  # Higher demand on weekends
        
        # Add monthly seasonality
        month_factors = {
            1: 0.8,  # January (post-holiday slump)
            2: 0.9,
            3: 1.0,
            4: 1.0,
            5: 1.1,
            6: 1.1,
            7: 1.2,  # Summer peak
            8: 1.2,
            9: 1.1,
            10: 1.0, 
            11: 1.2, # Holiday shopping begins
            12: 1.5  # Holiday peak
        }
        demand *= month_factors.get(self.month, 1.0)
        
        # Add some randomness
        demand *= np.random.normal(1.0, 0.05)  # 5% random noise
        
        return max(0, demand)  # Demand can't be negative
    
    def _calculate_reward(self, price_factor):
        """Calculate reward (profit) based on price and demand."""
        price = self.base_price * price_factor
        demand = self._calculate_demand(price_factor)
        
        # Simple cost model: fixed cost + variable cost per unit
        fixed_cost = self.base_price * 20  # Fixed cost equivalent to selling 20 units at base price
        unit_cost = self.base_price * 0.6  # 60% of base price is cost
        
        revenue = price * demand
        total_cost = fixed_cost + unit_cost * demand
        profit = revenue - total_cost
        
        # Normalize reward to be between -1 and 1 for stable learning
        normalized_reward = np.tanh(profit / 1000)
        
        return normalized_reward, profit, demand
    
    def step(self, action):
        """Take a step in the environment."""
        # Convert action to float, handling both array and scalar inputs
        price_factor = float(action) if np.isscalar(action) else float(action[0])
        
        # Ensure price factor is within bounds
        price_factor = np.clip(price_factor, self.price_range[0], self.price_range[1])
        
        # Set current price factor 
        self.current_price_factor = price_factor
        
        # Calculate reward, profit, and demand
        reward, profit, demand = self._calculate_reward(price_factor)
        
        # Update competitor price (simulate competitor behavior)
        # Competitors tend to follow successful pricing to some extent
        if profit > 0:
            # Competitor moves slightly toward our price
            self.competitor_price_factor = 0.95 * self.competitor_price_factor + 0.05 * price_factor
        else:
            # Competitor moves slightly away from our price
            diff = self.competitor_price_factor - price_factor
            self.competitor_price_factor += 0.05 * np.sign(diff)
        
        # Keep competitor price within bounds
        self.competitor_price_factor = np.clip(
            self.competitor_price_factor, self.price_range[0], self.price_range[1]
        )
        
        # Update calendar features (simulate passage of time)
        self.current_step += 1
        self.current_date += pd.Timedelta(days=1)
        self.day_of_week = self.current_date.weekday()
        self.month = self.current_date.month
        self.is_weekend = 1 if self.day_of_week >= 5 else 0
        
        # Create new observation
        observation = np.array([
            self.current_price_factor,
            self.competitor_price_factor,
            self.day_of_week,
            self.month,
            self.is_weekend
        ], dtype=np.float32)
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Include additional info
        info = {
            "profit": profit,
            "demand": demand,
            "price": self.base_price * price_factor,
            "base_price": self.base_price,
            "competitor_price": self.base_price * self.competitor_price_factor
        }

        return observation, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Current price factor: {self.current_price_factor:.2f}")
            print(f"Current price: ${self.base_price * self.current_price_factor:.2f}")
            print(f"Competitor price factor: {self.competitor_price_factor:.2f}")
            print(f"Competitor price: ${self.base_price * self.competitor_price_factor:.2f}")
            print(f"Day of week: {self.day_of_week} (Weekend: {self.is_weekend})")
            print(f"Month: {self.month}")
            print(f"Date: {self.current_date.strftime('%Y-%m-%d')}")
            _, profit, demand = self._calculate_reward(self.current_price_factor)
            print(f"Estimated demand: {demand:.2f} units")
            print(f"Estimated profit: ${profit:.2f}")
            print("-" * 50)
    
    def close(self):
        """Clean up resources."""
        pass


class PPONetwork(nn.Module):
    """Neural network for PPO algorithm."""
    
    def __init__(self, input_dim, hidden_dim, action_dim):
        """Initialize network architecture."""
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_std = nn.Parameter(torch.ones(action_dim) * 0.1)
        
        # Value head (critic)
        self.value = nn.Linear(hidden_dim, 1)
    
    # def forward(self, x):
    #     """Forward pass through network."""
    #     features = self.feature_extractor(x)
    #     action_mean = self.policy_mean(features)
    #     value = self.value(features)
        
    #     return action_mean, self.policy_std, value

    def forward(self, x):
        """Forward pass through network."""
        features = self.feature_extractor(x)
        
        # Add value checking
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
        
        action_mean = self.policy_mean(features)
        value = self.value(features)
        
        # Add value checking
        action_mean = torch.nan_to_num(action_mean, nan=0.0)
        value = torch.nan_to_num(value, nan=0.0)
        
        return action_mean, self.policy_std, value
    
    def get_action(self, state, action=None):
        """Sample an action from the policy distribution."""
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mean, action_std, value = self.forward(state)
        
        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Sample or use provided action
        if action is None:
            action = dist.sample()
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Calculate entropy
        entropy = dist.entropy().sum(dim=-1)
        
        return action.squeeze().detach().numpy(), value.squeeze().detach().numpy(), log_prob.detach(), entropy.detach()
    
    def evaluate(self, state, action):
        """Evaluate action given state."""
        action_mean, action_std, value = self.forward(state)
        
        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Calculate entropy
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value


class PPOAgent:
    """Proximal Policy Optimization agent for dynamic pricing."""
    
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 hidden_dim=128,
                 lr=0.001,
                 gamma=0.99,
                 epsilon=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 update_epochs=4,
                 mini_batch_size=64,
                 device="cpu"):
        """Initialize the PPO agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device
        
        # Initialize policy network
        self.network = PPONetwork(state_dim, hidden_dim, action_dim).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Initialize memory buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def select_action(self, state):
        """Select action based on current policy."""
        with torch.no_grad():
            action, value, log_prob, _ = self.network.get_action(state)

            # Ensure action is a scalar
            action = np.asarray(action).item()

        # Store experience in buffer
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        
        return action
    
    def store_outcome(self, reward, done):
        """Store reward and done flag in buffer."""
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
    
    def compute_advantages(self, next_value):
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        returns = []
        advantages = []
        
        gae = 0
        next_value = next_value
        
        # Reverse iteration for efficient advantage calculation
        for i in reversed(range(len(self.buffer['rewards']))):
            if i == len(self.buffer['rewards']) - 1:
                next_non_terminal = 1.0 - self.buffer['dones'][i]
                next_return = next_value
            else:
                next_non_terminal = 1.0 - self.buffer['dones'][i]
                next_return = returns[0]
            
            # Calculate return
            current_return = self.buffer['rewards'][i] + self.gamma * next_non_terminal * next_return
            returns.insert(0, current_return)
            
            # Calculate advantage
            delta = self.buffer['rewards'][i] + self.gamma * next_non_terminal * next_value - self.buffer['values'][i]
            gae = delta + self.gamma * 0.95 * next_non_terminal * gae
            advantages.insert(0, gae)
            
            next_value = self.buffer['values'][i]
        
        return returns, advantages
    
    def update_policy(self, next_value):
        # """Update policy using collected experiences."""
        # # Compute advantages and returns
        # returns, advantages = self.compute_advantages(next_value)
        
        # # Convert lists to tensors
        # states = torch.FloatTensor(self.buffer['states']).to(self.device)
        # actions = torch.FloatTensor(self.buffer['actions']).unsqueeze(1).to(self.device)
        # old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        # returns = torch.FloatTensor(returns).to(self.device)
        # advantages = torch.FloatTensor(advantages).to(self.device)

        """Update policy using collected experiences."""
        # Handle NaN next_value
        if np.isnan(next_value):
            next_value = 0.0
        
        # Compute advantages and returns
        returns, advantages = self.compute_advantages(next_value)
        
        # Convert lists to tensors and handle NaN values
        states = torch.FloatTensor(self.buffer['states']).to(self.device)
        actions = torch.FloatTensor(self.buffer['actions']).unsqueeze(1).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Handle NaN values
        states = torch.nan_to_num(states, nan=0.0)
        actions = torch.nan_to_num(actions, nan=0.0)
        old_log_probs = torch.nan_to_num(old_log_probs, nan=0.0)
        returns = torch.nan_to_num(returns, nan=0.0)
        advantages = torch.nan_to_num(advantages, nan=0.0)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimization loop
        for _ in range(self.update_epochs):
            # Create mini-batches
            indices = torch.randperm(states.size(0))
            for start_idx in range(0, states.size(0), self.mini_batch_size):
                # Get mini-batch indices
                idx = indices[start_idx:start_idx + self.mini_batch_size]
                
                # Evaluate actions and calculate ratio
                new_log_probs, entropy, values = self.network.evaluate(
                    states[idx], actions[idx]
                )
                
                # Calculate ratio of new and old probabilities
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                
                # Calculate surrogate losses
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages[idx]
                
                # Calculate final loss
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((values.squeeze() - returns[idx]) ** 2).mean()
                entropy_loss = -entropy.mean()
                
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear buffer after update
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def save(self, path):
        """Save model to file."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model from file."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class StaticPricingAgent:
    """Baseline static pricing agent for comparison."""
    
    def __init__(self, price_factor=1.0):
        """Initialize with a fixed price factor."""
        self.price_factor = price_factor
    
    def select_action(self, state):
        """Return static price factor regardless of state."""
        return np.array([self.price_factor])


class RulePricingAgent:
    """Rule-based pricing agent for comparison."""
    
    def __init__(self, base_price_factor=1.0, competitor_margin=0.05, weekend_factor=0.1):
        """Initialize with rule parameters."""
        self.base_price_factor = base_price_factor
        self.competitor_margin = competitor_margin
        self.weekend_factor = weekend_factor
    
    def select_action(self, state):
        """Apply pricing rules based on state."""
        # Extract state components
        _, competitor_price_factor, _, _, is_weekend = state
        
        # Basic competitor-based rule
        if competitor_price_factor > self.base_price_factor:
            # Competitor price is higher, we price slightly below
            price_factor = competitor_price_factor - self.competitor_margin
        else:
            # Competitor price is lower, we price slightly above min price
            price_factor = max(0.8, competitor_price_factor + self.competitor_margin)
        
        # Weekend discount rule
        if is_weekend:
            price_factor -= self.weekend_factor
        
        return np.array([price_factor])


#------------------------------------------------------------------------------
# Evaluation and Visualization Components
#------------------------------------------------------------------------------

class PricingEvaluator:
    """Evaluates and compares different pricing strategies."""
    
    def __init__(self, env, agents_dict, episodes=10):
        """Initialize with environment and agents to evaluate."""
        self.env = env
        self.agents = agents_dict
        self.episodes = episodes
        self.results = {}
    
    def evaluate_agent(self, agent_name):
        """Evaluate a single agent over multiple episodes."""
        agent = self.agents[agent_name]
        episode_rewards = []
        episode_profits = []
        episode_prices = []
        episode_demands = []
        
        for _ in range(self.episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_profit_list = []
            episode_price_list = []
            episode_demand_list = []
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_profit_list.append(info['profit'])
                episode_price_list.append(info['price'])
                episode_demand_list.append(info['demand'])
                
                state = next_state
                
                # For PPO agent, store outcome
                if hasattr(agent, 'store_outcome'):
                    agent.store_outcome(reward, done)
            
            # Update PPO agent if needed
            if hasattr(agent, 'update_policy'):
                # Get value estimate for the final state
                with torch.no_grad():
                    _, value, _, _ = agent.network.get_action(state)
                agent.update_policy(value)
            
            episode_rewards.append(episode_reward)
            episode_profits.append(sum(episode_profit_list))
            episode_prices.append(episode_price_list)
            episode_demands.append(episode_demand_list)
        
        # Calculate average results
        avg_reward = sum(episode_rewards) / self.episodes
        avg_profit = sum(episode_profits) / self.episodes
        
        # Store results
        self.results[agent_name] = {
            'avg_reward': avg_reward,
            'avg_profit': avg_profit,
            'all_rewards': episode_rewards,
            'all_profits': episode_profits,
            'all_prices': episode_prices,
            'all_demands': episode_demands
        }
        
        return avg_reward, avg_profit
    
    def evaluate_all_agents(self):
        """Evaluate all agents."""
        results = {}
        for agent_name in self.agents:
            print(f"Evaluating {agent_name}...")
            avg_reward, avg_profit = self.evaluate_agent(agent_name)
            results[agent_name] = {
                'avg_reward': avg_reward,
                'avg_profit': avg_profit
            }
            print(f"{agent_name}: Avg Reward = {avg_reward:.4f}, Avg Profit = {avg_profit:.2f}")
        
        return results
    
    def plot_price_comparison(self, episode_idx=0):
        """Plot price comparison for different agents."""
        plt.figure(figsize=(12, 6))
        
        for agent_name in self.results:
            prices = self.results[agent_name]['all_prices'][episode_idx]
            plt.plot(prices, label=f"{agent_name}")
        
        plt.title("Price Comparison Across Strategies")
        plt.xlabel("Time Step")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.savefig("price_comparison.png")
        plt.close()
    
    def plot_profit_comparison(self, episode_idx=0):
        """Plot cumulative profit comparison for different agents."""
        plt.figure(figsize=(12, 6))
        
        for agent_name in self.results:
            profits = self.results[agent_name]['all_profits']
            plt.bar(agent_name, profits[episode_idx], alpha=0.7)
        
        plt.title("Total Profit Comparison")
        plt.xlabel("Agent")
        plt.ylabel("Total Profit ($)")
        plt.grid(True, axis='y')
        plt.savefig("profit_comparison.png")
        plt.close()
    
    def plot_demand_vs_price(self, agent_name, episode_idx=0):
        """Plot demand vs price for a specific agent."""
        plt.figure(figsize=(12, 6))
        
        prices = self.results[agent_name]['all_prices'][episode_idx]
        demands = self.results[agent_name]['all_demands'][episode_idx]
        
        plt.scatter(prices, demands, alpha=0.7)
        plt.title(f"Demand vs Price for {agent_name}")
        plt.xlabel("Price ($)")
        plt.ylabel("Demand (units)")
        plt.grid(True)
        plt.savefig(f"demand_vs_price_{agent_name}.png")
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive evaluation report."""
        report = {
            'summary': {},
            'detailed': self.results
        }
        
        # Calculate summary statistics
        for agent_name in self.results:
            report['summary'][agent_name] = {
                'avg_reward': self.results[agent_name]['avg_reward'],
                'avg_profit': self.results[agent_name]['avg_profit'],
                'profit_std': np.std([p for p in self.results[agent_name]['all_profits']]),
                'max_profit': max(self.results[agent_name]['all_profits']),
                'min_profit': min(self.results[agent_name]['all_profits'])
            }
        
        # Find best agent
        best_agent = max(report['summary'], key=lambda x: report['summary'][x]['avg_profit'])
        report['best_agent'] = best_agent
        
        return report


#------------------------------------------------------------------------------
# Main System Integration
#------------------------------------------------------------------------------

class DynamicPricingSystem:
    """Main system that integrates all components."""
    
    def __init__(self, 
                 product_urls=None,
                 data_file="product_data.csv", 
                 model_dir="models",
                 use_scraping=True):
        """Initialize the dynamic pricing system."""
        self.product_urls = product_urls or {}
        self.data_file = data_file
        self.model_dir = model_dir
        self.use_scraping = use_scraping
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Initialize components
        if self.use_scraping:
            self.scraper_manager = ScraperManager()
        self.data_processor = DataProcessor(self.data_file)
        
        # Variables to be initialized later
        self.env = None
        self.ppo_agent = None
        self.static_agent = None
        self.rule_agent = None
        self.evaluator = None
    
    def collect_data(self):
        """Collect data from e-commerce platforms."""
        if not self.use_scraping:
            print("Skipping data collection (scraping disabled)")
            return
        
        print("Collecting data from e-commerce platforms...")
        self.scraper_manager.scrape_product_across_platforms(self.product_urls)
    
    # def prepare_environment(self):
    #     """Prepare the RL environment using collected data."""
    #     # Load and process data
    #     df = self.data_processor.load_data()
    #     if df is None or df.empty:
    #         raise ValueError("No data available. Run collect_data first.")
        
    #     df_clean = self.data_processor.clean_data(df)
        
    #     # Create environment
    #     self.env = PricingEnvironment(df_clean)
        
    #     print(f"Environment prepared with base price: ${self.env.base_price:.2f}")
        
    #     return self.env
    
    def prepare_environment(self):
        """Prepare the RL environment using collected data."""
        # Load and process data
        df = self.data_processor.load_data()
        if df is None or df.empty:
            raise ValueError("No data available. Run collect_data first.")
        
        df_clean = self.data_processor.clean_data(df)
        
        # Ensure we have valid price data
        if df_clean['price'].isna().all():
            # If all prices are NaN, use a default base price
            df_clean['price'] = 100.0  # Default price
        else:
            # Fill NaN prices with median
            df_clean['price'] = df_clean['price'].fillna(df_clean['price'].median())
        
        # Create environment
        self.env = PricingEnvironment(df_clean)
        
        if np.isnan(self.env.base_price):
            self.env.base_price = 100.0  # Default base price if NaN
        
        print(f"Environment prepared with base price: ${self.env.base_price:.2f}")
        
        return self.env

    def setup_agents(self):
        """Set up different pricing agents."""
        if self.env is None:
            raise ValueError("Environment not prepared. Run prepare_environment first.")
        
        # Set up PPO agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.ppo_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=0.001,
            gamma=0.99,
            epsilon=0.2
        )
        
        # Set up static pricing agent (at base price)
        self.static_agent = StaticPricingAgent(price_factor=1.0)
        
        # Set up rule-based pricing agent
        self.rule_agent = RulePricingAgent(
            base_price_factor=1.0,
            competitor_margin=0.05,
            weekend_factor=0.1
        )
        
        # Set up evaluator
        self.evaluator = PricingEvaluator(
            env=self.env,
            agents_dict={
                'PPO': self.ppo_agent,
                'Static': self.static_agent,
                'Rule-Based': self.rule_agent
            },
            episodes=10
        )
        
        print("Agents set up successfully")
    
    def train_agent(self, num_episodes=100):
        """Train the RL agent."""
        if self.ppo_agent is None:
            raise ValueError("Agents not set up. Run setup_agents first.")
        
        print(f"Training PPO agent for {num_episodes} episodes...")
        
        # Training loop
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Select action
                action = self.ppo_agent.select_action(state)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Store outcome
                self.ppo_agent.store_outcome(reward, done)
                
                # Update state and accumulate reward
                state = next_state
                episode_reward += reward
            
            # Update policy after episode
            with torch.no_grad():
                _, value, _, _ = self.ppo_agent.network.get_action(state)
            self.ppo_agent.update_policy(value)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.4f}")
        
        # Save trained model
        model_path = os.path.join(self.model_dir, "ppo_agent.pt")
        self.ppo_agent.save(model_path)
        print(f"Model saved to {model_path}")
    
    def evaluate_strategies(self):
        """Evaluate different pricing strategies."""
        if self.evaluator is None:
            raise ValueError("Agents not set up. Run setup_agents first.")
        
        print("Evaluating pricing strategies...")
        results = self.evaluator.evaluate_all_agents()
        
        # Plotting
        self.evaluator.plot_price_comparison()
        self.evaluator.plot_profit_comparison()
        
        for agent_name in results:
            self.evaluator.plot_demand_vs_price(agent_name)
        
        # Generate report
        report = self.evaluator.generate_report()
        print("\nEvaluation Report:")
        print("-" * 50)
        print(f"Best Agent: {report['best_agent']}")
        for agent_name, stats in report['summary'].items():
            print(f"{agent_name}:")
            print(f"  Avg Profit: ${stats['avg_profit']:.2f}")
            print(f"  Profit Range: ${stats['min_profit']:.2f} - ${stats['max_profit']:.2f}")
        
        return report
    
    def deploy_optimal_strategy(self, test_states=None):
        """Deploy the optimal pricing strategy."""
        if self.evaluator is None or not self.evaluator.results:
            raise ValueError("Evaluation not performed. Run evaluate_strategies first.")
        
        # Get the best agent based on evaluation
        report = self.evaluator.generate_report()
        best_agent_name = report['best_agent']
        best_agent = self.evaluator.agents[best_agent_name]
        
        print(f"Deploying optimal strategy: {best_agent_name}")
        
        # Test on sample market conditions
        if test_states is None:
            # Create some test market conditions
            test_states = [
                np.array([1.0, 0.9, 1, 6, 0]),  # Weekday in June, competitor price lower
                np.array([1.0, 1.1, 6, 6, 1]),  # Weekend in June, competitor price higher
                np.array([1.0, 1.0, 3, 12, 0]), # Weekday in December, same competitor price
                np.array([1.0, 0.8, 5, 12, 1])  # Weekend in December, competitor price much lower
            ]
        
        # Get optimal prices for each condition
        results = []
        for i, state in enumerate(test_states):
            action = best_agent.select_action(state)
            # Handle action whether it's a float or array
            price_factor = action if isinstance(action, float) else action[0]
            price = self.env.base_price * price_factor
            
            # Get state description
            day_type = "Weekend" if state[4] == 1 else "Weekday"
            month = int(state[3])
            competitor_price = self.env.base_price * state[1]
            
            result = {
                "condition": f"{day_type} in month {month}, competitor price: ${competitor_price:.2f}",
                "optimal_price": price,
                "price_factor": price_factor
            }
            results.append(result)
            
            print(f"\nMarket Condition {i+1}: {result['condition']}")
            print(f"  Optimal Price: ${result['optimal_price']:.2f}")
            print(f"  (Factor: {result['price_factor']:.2f}x base price)")
        
        return results
    
    # def close(self):
    #     """Clean up resources."""
    #     if hasattr(self, 'scraper_manager') and self.scraper_manager:
    #         self.scraper_manager.close()
    #     if hasattr(self, 'env') and self.env:
    #         self.env.close()

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'scraper_manager') and self.scraper_manager:
            self.scraper_manager.close()
        if hasattr(self, 'env') and self.env:
            self.env.close()
        if hasattr(self, 'ppo_agent') and self.ppo_agent:
            self.ppo_agent.network.cpu()
            del self.ppo_agent
        if hasattr(self, 'static_agent') and self.static_agent:
            del self.static_agent
        if hasattr(self, 'rule_agent') and self.rule_agent:
            del self.rule_agent
        if hasattr(self, 'evaluator') and self.evaluator:
            del self.evaluator
        print("Resources cleaned up successfully.")
        
#------------------------------------------------------------------------------
# Example Usage
#------------------------------------------------------------------------------

# def main():
#     """Main function to demonstrate the system."""
#     # Define product URLs
#     product_urls = {
#         "amazon": "https://www.amazon.in/dp/B0DGJ7TGDR/?_encoding=UTF8&ref_=cct_cg_Budget_2c1",  # Example product
#         "flipkart": "https://www.flipkart.com/apple-iphone-16-white-128-gb/p/itm7c0281cd247be"  # Example product
#     }


def main():
    """Main function to demonstrate the system."""
    # Get product URLs from user
    print("Enter product URLs (leave blank and press Enter when done):")
    print("Format: platform url")
    print("Example: amazon https://www.amazon.com/product")
    
    product_urls = {}
    while True:
        try:
            entry = input("\nEnter platform and URL (or press Enter to finish): ").strip()
            if not entry:
                break
                
            platform, url = entry.split(maxsplit=1)
            platform = platform.lower()
            
            if platform not in ['amazon', 'flipkart']:
                print("Error: Platform must be 'amazon' or 'flipkart'")
                continue
                
            product_urls[platform] = url.strip()
            
        except ValueError:
            print("Error: Please enter both platform and URL")
            continue
    
    if not product_urls:
        print("No URLs provided. Using default test URLs...")
        product_urls = {
            "amazon": "https://www.amazon.in/dp/B0DGJ7TGDR/?_encoding=UTF8&ref_=cct_cg_Budget_2c1",
            "flipkart": "https://www.flipkart.com/apple-iphone-16-white-128-gb/p/itm7c0281cd247be"
        }
    
    print("\nCollecting data for the following URLs:")
    for platform, url in product_urls.items():
        print(f"{platform}: {url}")
    
    # Initialize system
    system = DynamicPricingSystem(
        product_urls=product_urls,
        data_file="product_data.csv",
        model_dir="models",
        use_scraping=True
    )
    
    try:
        # Step 1: Collect data
        system.collect_data()
        
        # Step 2: Prepare environment
        system.prepare_environment()
        
        # Step 3: Set up agents
        system.setup_agents()
        
        # Step 4: Train RL agent
        system.train_agent(num_episodes=100)
        
        # Step 5: Evaluate strategies
        system.evaluate_strategies()
        
        # Step 6: Deploy optimal strategy
        system.deploy_optimal_strategy()
    
    finally:
        # Clean up
        system.close()
        print("System closed successfully.")

if __name__ == "__main__":
    main()
# 🤖 Dynamic Pricing System using Reinforcement Learning & Web Scraping

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-success)
![RL](https://img.shields.io/badge/Reinforcement--Learning-PPO-purple)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Welcome to the **Dynamic Pricing System**, an AI-powered platform that optimizes product pricing using **Reinforcement Learning** and **real-time web scraping** from e-commerce sites like **Amazon** and **Flipkart**. 🛒📈

---

## 🚀 What is this project about?

This system automatically:
- Scrapes competitor prices from live e-commerce websites
- Simulates market conditions using OpenAI Gym
- Trains a **PPO (Proximal Policy Optimization)** agent to learn pricing strategies
- Evaluates and compares with static and rule-based agents
- Recommends **optimal pricing** for maximum profitability 💰

---

## 🎯 Key Features

👉 Real-time Web Scraping  
👉 PPO-based Reinforcement Learning Model  
👉 Market Simulation Environment  
👉 Evaluation of Multiple Pricing Strategies  
👉 Streamlit UI for Interaction & Visualization  
👉 Deployment of Optimal Pricing Recommendations

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** – Interactive Web UI
- **Selenium + BeautifulSoup** – Web Scraping
- **PyTorch + OpenAI Gym** – Reinforcement Learning
- **Pandas, NumPy** – Data Processing
- **Plotly, Matplotlib** – Visualizations

---

## 📸 Preview

> ✨ Here are a few screenshots of the interface:

![Price Comparison](price_comparison.png)  
![Profit Analysis](profit_comparison.png)  
![Agent Performance](demand_vs_price_PPO.png)

---

## ⚙️ How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/dynamic-pricing-rl.git
   cd dynamic-pricing-rl
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

4. **Enter Product URLs** from Amazon or Flipkart and follow the on-screen steps.

---

## 📂 Project Structure

```
.
├── app.py                     # Streamlit app interface
├── main.py                    # Core logic: scraping, training, evaluation
├── product_data.csv           # Collected data
├── models/                    # Saved PPO models
├── screenshots/               # Output visualizations
└── README.md
```

---

## 📌 Future Enhancements

- Add support for more e-commerce platforms
- Improve demand forecasting using real sales data
- Integrate real-time deployment APIs
- Add authentication and cloud deployment support

---

## 🤝 Contributing

We welcome contributions! Feel free to fork this repo, create a branch, and submit a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## ✨ Created by

**S Manikanta** | **Y Bhavya** | **K Rohan**  
*Department of CSE (Data Science), NNRG Institutions*


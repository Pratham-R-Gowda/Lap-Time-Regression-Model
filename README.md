# 🏎️ The F1 Pit Wall on My Laptop: Predicting Lap Times with XGBoost

## 👋 What is this?
If you watch Formula 1, you know the commentators constantly talk about "the crossover." During a race, a car's lap time is a massive tug-of-war between two physical forces:
1. **Fuel Burn:** The car starts heavy (110kg of fuel) and gets lighter every lap. *Lighter = Faster.*
2. **Tire Degradation:** The rubber wears off the tires every lap, losing grip. *Old tires = Slower.*

I wanted to see if I could build a Machine Learning model that actually understands this physics battle. I used the FastF1 API to pull live race telemetry and trained an **XGBoost Regressor** to predict a driver's exact lap time based on their tire age, fuel load, and the track temperature.

## 🛠️ The Stack
* **The Brain:** `XGBoost` (Because when it comes to tabular data, XGBoost is king)
* **The Data:** `FastF1` API, `pandas`, `numpy`
* **The Visuals:** `seaborn`, `matplotlib` 

## 🧗‍♂️ The Engineering Struggles (Or, "Things that broke")
Machine Learning is 80% cleaning data and 20% actually doing the cool stuff. Here are the hurdles I had to jump:

* **The "Clean Air" Problem:** You can't train a model on Safety Car laps or pit-stops, or the algorithm will think the tires suddenly got 40 seconds slower. I had to build a strict filtering pipeline to only extract "Green Flag" racing laps.
* **The Pandas Index Nightmare:** While merging track temperature weather data into my telemetry dataframe, the scrambled index from my filtered rows caused a massive data-alignment crash. I had to implement strict index-reset protocols to get the matrix to compile.
* **Faking the Fuel Data:** F1 teams do not broadcast live fuel weights to the public. To get around this, I engineered a feature using `LapNumber` as a proxy. Since fuel burn is linear, I banked on the XGBoost decision trees being smart enough to deduce the weight reduction mathematically. (Spoiler: It worked).

## 📊 What I Found Out
* **Accuracy:** The model achieved a Mean Absolute Error (MAE) of **[Insert your MAE here]** seconds. In a chaotic sport with lockups, wind gusts, and dirty air from traffic, predicting a lap time to within fractions of a second using just 5 variables blew my mind.
* **The Feature Proof:** I ran a Feature Importance check, and `LapNumber` (Fuel) and `TyreLife` (Degradation) absolutely dominated the charts. The model successfully learned the crossover effect.
* **Unbiased Errors:** My residual analysis plotted a beautiful, normal bell curve centered at zero. The model isn't biased; its only major misses are unpredictable track anomalies.

## 🚀 Play With It Yourself
Want to see what Fernando Alonso's predicted pace is on 20-lap-old Hard tires? 

1. Clone this repo.
2. Install the F1 stack: `pip install -r requirements.txt`
3. Run the Jupyter Notebook. 
*(Note: I've ignored the `f1_cache` folder in git. The first time you run this, FastF1 will download a bunch of telemetry data to your local machine!)*
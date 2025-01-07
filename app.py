from flask import Flask, render_template, request, url_for
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import os
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = None
    if request.method == "POST":
        # Load the dataset
        data = pd.read_excel("data/google_dataset.xlsx")

        # Process and analyze the data
        data['Month Starting'] = pd.to_datetime(data['Month Starting'], errors='coerce')  # Ensure datetime conversion
        data_cleaned = data.dropna(subset=['Month Starting'])
        data['Returns'] = data['Close'].pct_change()
        data['Rolling Average'] = data['Returns'].rolling(window=30).mean()
        data = data.dropna(subset=['Returns', 'Rolling Average'])

        # Scale the Returns
        scaler = StandardScaler()
        data['Returns'] = scaler.fit_transform(data['Returns'].values.reshape(-1, 1))

        # Anomaly Detection using IsolationForest
        iso_forest = IsolationForest(contamination=0.05)
        data['Anomaly_IF'] = iso_forest.fit_predict(data[['Returns']])
        data['Anomaly_IF'] = data['Anomaly_IF'].apply(lambda x: 1 if x == -1 else 0)

        # Anomaly Detection using DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        data['Anomaly_DB'] = dbscan.fit_predict(data[['Returns']])
        data['Anomaly_DB'] = data['Anomaly_DB'].apply(lambda x: 1 if x == -1 else 0)

        # Anomaly Detection using LOF (Local Outlier Factor)
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        data['Anomaly_LOF'] = lof.fit_predict(data[['Returns']])
        data['Anomaly_LOF'] = data['Anomaly_LOF'].apply(lambda x: 1 if x == -1 else 0)

        # Plotting
        plt.figure(figsize=(15, 5))

        # Plot IsolationForest anomalies
        plt.subplot(131)
        plt.plot(data['Month Starting'], data['Returns'], label='Returns')
        plt.scatter(data[data['Anomaly_IF'] == 1]['Month Starting'], 
                    data[data['Anomaly_IF'] == 1]['Returns'], color='red', label='Anomalies')
        plt.title('Isolation Forest Anomalies')

        # Plot DBSCAN anomalies
        plt.subplot(132)
        plt.plot(data['Month Starting'], data['Returns'], label='Returns')
        plt.scatter(data[data['Anomaly_DB'] == 1]['Month Starting'], 
                    data[data['Anomaly_DB'] == 1]['Returns'], color='red', label='Anomalies')
        plt.title('DBSCAN Anomalies')

        # Plot LOF anomalies
        plt.subplot(133)
        plt.plot(data['Month Starting'], data['Returns'], label='Returns')
        plt.scatter(data[data['Anomaly_LOF'] == 1]['Month Starting'], 
                    data[data['Anomaly_LOF'] == 1]['Returns'], color='red', label='Anomalies')
        plt.title('LOF Anomalies')

        plt.tight_layout()


        

        # Ensure the 'static' directory exists
        static_dir = os.path.join(os.getcwd(), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        # Now save the plot
        plt.savefig(os.path.join(static_dir, "anomaly_detection_plots.png"))

        plt.close()

        # Create the URL for the plot image
        plot_url = url_for("static", filename="anomaly_detection_plots.png")

    return render_template("index.html", plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)

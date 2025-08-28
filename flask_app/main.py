from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib
# Use non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import wandb
import os
from dotenv import load_dotenv
import json
from transformers import RobertaForSequenceClassification, AutoTokenizer, pipeline


app = Flask(__name__)
CORS(app)

COLORS = ["#D17378", "#C44A51", "#260D0E"]


def login():
    """Retrieves W&B API and automatically logs in"""
    try:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=wandb_api_key)

    except Exception as e:
        raise

def model_download():
    """Retrieve trained model from W&B."""
    try:
        # Initialize a W&B run
        run = wandb.init(project="roberta-youtube-analyzer")

        # Retrieve model and download it
        artifact_name = os.getenv("FINAL_MODEL")
        artifact = run.use_artifact(artifact_name, type="model")
        model_path = artifact.download()

        # Load model
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer

    except Exception as e:
        raise



# Login to W&B
login()

# Initialize model and tokenizer
model, tokenizer = model_download()


# URL triggers
@app.route("/")
def home():
    return "Welcome!"


@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    """Retrieves comments and timestamps and returns predicted sentiment with timestamps."""
    # Get comments
    data = request.json
    comments_data = data.get("comments")

    # Send response in case comments are missing
    if not comments_data:
        return jsonify({"error": "No comments were found."}, status=400)

    try:
        # Separate comments and timestamps
        comments = [item["text"] for item in comments_data]
        timestamps = [item["timestamp"] for item in comments_data]

        # Instantiate a classifier pipeline to make prei
        classifier = pipeline(
            "sentiment-analysis", 
            model=model,
            tokenizer=tokenizer,
            device=-1,
            return_all_scores=True,
            )

        # Extract sentiment label from predictions
        preds = [dict["label"] for dict in classifier(comments)]
        

    except Exception as e:
        app.logger.error(f"Error in /predict_with_timestamps: {e}")
        return jsonify({"error": f"Couldn't make a prediction: {str(e)}"}, status=500)

    # Return the response
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, preds, timestamps)]
    return jsonify(response)


@app.route("/predict", methods=["POST"])
def predict():
    """Retrieves comments and returns predicted sentiment."""
    # Get comments
    data = request.json
    comments_data = data.get("comments")

    # Send response in case comments are missing
    if not comments_data:
        return jsonify({"error": "No comments were found."}, status=400)

    try:
        # Extract comments
        comments = [item["text"] for item in comments_data]

        # Instantiate a classifier pipeline to make prei
        classifier = pipeline(
            "sentiment-analysis", 
            model=model,
            tokenizer=tokenizer,
            device=-1,
            return_all_scores=True,
            )

        # Extract sentiment label from predictions
        preds = [dict["label"] for dict in classifier(comments)]
        

    except Exception as e:
        app.logger.error(f"Error in /predict: {e}")
        return jsonify({"error": f"Couldn't make a prediction: {str(e)}"}, status=500)

    # Return the response
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, preds)]
    return jsonify(response)


@app.route("/chart", methods=["POST"])
def generate_chart():
    """Generates pie chart from sentiment data."""
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts")

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts were provided."}, status=400)

        # Prepare data
        labels = ["Positive", "Neutral", "Negative"]
        counts = [
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("2", 0))
        ]

        # Raise error if all counts are zero
        if sum(counts) == 0:
            raise ValueError("Sentiment counts sum to zero.")
        
        # Generate pie chart
        plt.figure(figsize=(6,6))
        plt.pie(
            counts,
            explode=(0, 0, 0.1),
            labels=labels,
            colors=COLORS,
            autopct="%1.1f%%",
            shadow=True,
            textprops=dict(color="w")
        )
        plt.axis("equal")

        # Save the chart as BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        # Return as respomse
        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        app.logger.error(f"Error in /chart: {e}")
        return jsonify({"error": f"Couldn't generate the chart: {str(e)}"}, status=500)


@app.route("/wordcloud", methods=["POST"])
def generate_wordcloud():
    """Generate wordcloud from comments."""
    try:
        data = request.get_json()
        comments = data.get("comments")

        if not comments:
            return jsonify({"error": "No comments were provided."}, status=400)

        text = " ".join(comments)

        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            backgroud_color=None,
            mode="RGBA",
            colormap="gist_yang",
            collocations=False
        ).generate(text)

        # Save word cloud as BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        # Return as response
        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        app.logger.error(f"Error in /wordcloud: {e}")
        return jsonify({"error": f"Couldn't generate wordcloud: {e}"}, status=500)


@app.route("/trend_graph", methods=["POST"])
def generate_trend_graph():
    """Generate a graph showing trends in sentiment."""
    try:
        data = request.get_json()
        sentiment_data = data.get("sentiment_data")

        if not sentiment_data:
            return jsonify({"error": "No sentiment data were provided."}, status=400)

        # Convert to pandas DataFrame
        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)

        # Map sentiment values to labels
        df["sentiment"] = df["sentiment"].astype(int)
        sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

        # Downsample data into 1 month bins and count sentiment values occurrences
        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)

        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [0, 1, 2]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by value
        monthly_percentages = monthly_percentages[[0, 1, 2]]

        # Plot trends: x = timestamp, y = sentiment percentage
        plt.figure(figsize=(12, 6))

        for sentiment_value in [0, 1, 2]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker=".",
                linestyle="-",
                color=COLORS[sentiment_value]
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        # Save as BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        # Return as response
        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        app.logger.error(f"Error in /trend_graph: {e}")
        return jsonify({"error": f"Couldn't generate trend_graph: {e}"}, status=500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
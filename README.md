# roberta-youtube-analyzer
Sentiment analysis on YouTube comments using RoBERTa Transfomers and MLOps best practices.

## 🙏 Acknowledgments

This project was inspired by the excellent tutorial by [entbappy](https://github.com/entbappy) on [End-to-end Youtube Sentiment Analysis](https://github.com/entbappy/End-to-end-Youtube-Sentiment). While I've significantly modified the approach by using RoBERTa Transformers instead of traditional ML models, Kaggle for training and a different dataset, the original tutorial provided valuable insights into structuring an MLOps pipeline for sentiment analysis.

### Key differences from the original:
- **Model**: RoBERTa-base fine-tuned vs. scikit-learn model
- **Training Environment**: Kaggle cloud training vs. local training  
- **Dataset**: YouTube comments dataset vs. Reddit dataset
- **MLOps**: Added W&B experiment tracking instead of MLFlow

# 🧮 Credit Score Classification Project

---

## 📊 **Project Overview**

The primary objective of this project was to develop a robust machine learning model to accurately classify credit scores into **Good**, **Standard**, and **Poor** categories based on diverse financial and demographic features. Through a structured data science approach involving **data wrangling**, **feature selection**, **model optimization**, and **evaluation**, the final model achieved a **high accuracy of 85.33%** on the **test set**.

---

## 🔍 **Key Findings**

### 1. **Model Performance**
- The **RandomForestClassifier** with **optimized parameters** provided the best results, outperforming the initial baseline models.
- Demonstrated strong **generalization ability** with:
  - **Cross-validation accuracy:** 84.25%
  - **Test set accuracy:** 85.33%

### 2. **Feature Importance**
- The most influential features in determining **credit scores** included:
  - **Annual Income**
  - **Credit History Age**
  - **Outstanding Debt**
- Feature analysis provided insights into **key financial behaviors** contributing to **creditworthiness**.

### 3. **Model Stability**
- The **confusion matrix** and **classification report** showed balanced performance across all **credit score categories**, with **high precision and recall**.

### 4. **Insights from Visualizations**
- Individuals with **higher incomes** and **longer credit histories** were more likely to have **good credit scores**.
- The **number of loans** and **credit utilization ratio** also played significant roles in **credit score classification**.

### 5. **Test Case Predictions**
- The model was tested with **edge case scenarios**, demonstrating reliable predictions with custom **decision thresholds**:
- High-income, low-debt scenarios accurately predicted as **Credit_Score_Standard**.
- Low-income, high-debt scenarios correctly identified as **Credit_Score_Poor**.
- The thresholds addition minimized **overlap** in predictions, enhancing **classification accuracy**.

---

## ✅ **Conclusion**

The final model not only met but **exceeded expectations** by achieving **strong predictive accuracy** and **model stability**. The insights derived from the **model's performance metrics** and **feature analysis** provide a **solid foundation** for **credit risk management strategies**. This model can support **financial institutions** in **enhancing decision-making**, **mitigating risks**, and **promoting responsible lending practices**.

---

## 🚀 **Next Steps**

- **Deployment:** The model is ready to be **integrated into production environments**, potentially through a **Flask API** or **automated dashboard**.
- **Real-World Impact:** By offering **predictive insights**, this model can contribute to **financial stability** and support individuals in improving their **credit health**.

---

## 💡 **Feedback & Contributions**

I appreciate any **feedback** and **suggestions** that could drive this project to its **next phase of development**. Feel free to **open issues** or **submit pull requests** to enhance this project.

---

### 🛠️ **Technologies Used**

- **Python**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Machine Learning**: RandomForestClassifier, LogisticRegression, SupportVectorMachine, GridSearchCV, RandomizedSearchCV
- **Data Visualization**: Seaborn, Matplotlib

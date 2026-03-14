# Federated-Machine-Learning-for-Predicting-Animal-to-Human-Zoonotic-Disease-Outbreaks
This project implements a hybrid federated deep neural network (FDNN) combined with XGBoost to predict zoonotic disease outbreaks across distributed datasets (Farm, Hospital, Wildlife) without sharing raw data. Includes EDA, preprocessing, model training, and a Flask-based web dashboard for visualization and predictions.

## Features

- **Federated Learning**: Train machine learning models across distributed datasets (Farm, Hospital, Wildlife) without sharing raw data.
- **Web Dashboard**: Interactive dashboard for monitoring predictions and historical data.
- **Model Management**: Save and load global and local models (FDNN - Federated Deep Neural Network).
- **Data Upload**: Support for uploading datasets for analysis.
- **User Authentication**: Login and registration system for secure access.

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Keras (for FDNN models)
- **Frontend**: HTML, CSS (via Jinja2 templates)
- **Data Processing**: Pandas, NumPy
- **Database**: (If applicable, e.g., SQLite or specify)
- **Environment**: Conda (aienv)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dayavallepu/Federated-Machine-Learning-for-Predicting-Animal-to-Human-Zoonotic-Disease-Outbreaks.git
   cd Federated-Machine-Learning-for-Predicting-Animal-to-Human-Zoonotic-Disease-Outbreaks
   ```

2. **Set up the Python environment**:
   - Ensure you have Conda installed.
   - Create and activate the environment:
     ```bash
     conda create -n aienv python=3.8
     conda activate aienv
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare datasets**:
   - Place your CSV datasets in the `Datasets/` folder (e.g., `Farm_data.csv`, `Hospital_data.csv`, `Wildlife_data.csv`).

5. **Run the application**:
   ```bash
   python app.py
   ```
   - Access the app at `http://localhost:5000` (default Flask port).

## Exploratory Data Analysis (EDA)

The EDA process is detailed in the Jupyter notebooks located in the `EDA/` folder:

- `EDA Farm.ipynb`: Analyzes farm-related dataset.
- `EDA Hospital.ipynb`: Analyzes hospital-related dataset.
- `EDA Wildlife.ipynb`: Analyzes wildlife-related dataset.

### EDA Steps:
1. **Load Dataset & Initial Inspection**:
   - Load CSV files (e.g., `Farm_data.csv`, `Hospital_data.csv`, `Wildlife_data.csv`).
   - Check dataset shape, preview data, and data types.
   - Verify suitability for federated learning.

2. **Data Quality Checks**:
   - Check for duplicates.
   - Identify null/missing values.

3. **Target Variable Analysis**:
   - Analyze distribution of the target variable (Outbreak Risk Levels: 0=Low, 1=Medium, 2=High).
   - Visualize using count plots.

4. **Missing Values Visualization**:
   - Heatmap to show patterns of missing data.

5. **Client-Specific EDA**:
   - Boxplots and violin plots for feature distributions by risk level.
   - Correlation matrices to identify relationships.
   - Feature importance via correlation with target.

6. **Automated EDA**:
   - Use Sweetviz for comprehensive reports.

7. **Distribution Analysis**:
   - Histograms for each feature.
   - Q-Q plots to check normality.
   - Scatter plots vs. target variable.

8. **Summary Insights**:
   - Key findings per client (e.g., strong correlations in specific features).

## Data Preprocessing

Preprocessing is implemented in the `Data Preprocessing and Model Building/Preprocessing and Model Building.ipynb` notebook.

### Preprocessing Steps:
1. **Outlier Capping**:
   - Use IQR method to cap outliers (factor=1.5).

2. **Missing Value Imputation**:
   - Impute missing values using median strategy.

3. **Feature Scaling**:
   - Apply RobustScaler for normalization.

4. **Feature Alignment**:
   - Align client-specific features to a global feature set for federated learning.

5. **Visualization**:
   - Plot distributions, missing values, and boxplots before/after preprocessing.

## Model Building

Model development is covered in the `Data Preprocessing and Model Building/Preprocessing and Model Building.ipynb` notebook.

### Model Architecture:
- **Hybrid Model**: Federated Deep Neural Network (FDNN) + XGBoost.
- FDNN: 3-layer dense network (64, 32, 3 units) with ReLU and softmax.

### Training Process:
1. **Local Model Training**:
   - Train FDNN on each client dataset (Farm, Hospital, Wildlife).
   - Use train-test split (80-20) with stratification.
   - Evaluate with accuracy and classification report.

2. **Federated Averaging (FedAvg)**:
   - Aggregate local model weights to create a global FDNN model.

3. **Feature Extraction**:
   - Use global FDNN to extract features from the second-last layer.

4. **XGBoost Training**:
   - Train XGBoost classifier on extracted features.
   - Parameters: n_estimators=300, max_depth=6, learning_rate=0.05.

5. **Final Evaluation**:
   - Assess combined model performance.
   - Generate confusion matrix.

6. **Model Saving**:
   - Save local models in `saved_local_models/`.
   - Save global model in `saved_global_model/`.
   - Export features for XGBoost as CSV.

## Usage

- **Login/Register**: Create an account or log in to access the dashboard.
- **Dashboard**: View predictions, upload data, and monitor model performance.
- **Model Training**: Trigger federated training rounds (ensure models are saved in `saved_local_models/` and `saved_global_model/`).
- **History**: Review past predictions and outbreaks.

## Project Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Datasets/              # Sample datasets (CSV files)
├── saved_global_model/    # Global federated model
├── saved_local_models/    # Local models per data source
├── templates/             # HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── index.html
│   └── ...
└── uploads/               # Uploaded files
```

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

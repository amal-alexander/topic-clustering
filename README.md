# Blog Topic Duplicate Detector

A Streamlit web app for **AI-powered detection and grouping of duplicate or similar blog topics**. This tool helps content creators, marketers, and SEO specialists organize, deduplicate, and optimize blog topic ideas for a more effective content strategy.

## Features

- 📝 **Input Topics**: Paste your blog topics (one per line) or upload a CSV file with your content ideas.
- 🎯 **Exact Duplicate Detection**: Finds topics that are textual duplicates.
- 🔍 **Near-Duplicate Identification**: Groups topics that are similar in meaning using NLP and clustering.
- 📊 **Visual Similarity Analysis**: Interactive similarity matrix and distribution charts.
- 💡 **Smart Recommendations**: Suggestions on merging or rewriting similar topics.
- 📥 **CSV Export Functionality**: Download your grouped and deduplicated topics for further use.
- 🎨 **Interactive Visualizations**: Explore your data with heatmaps and charts.

## How It Works

1. **Input Topics**:  
   - Paste your blog topics in the text area (one per line), _or_
   - Upload a CSV file with a `topic` or `title` column.
2. **AI Analysis**:  
   - The app preprocesses the topics (lowercasing, cleaning, stop words removal).
   - Calculates similarity using TF-IDF and cosine similarity.
   - Clusters similar topics using DBSCAN.
   - Detects exact and near-duplicate groups.
3. **Get Results**:  
   - View analysis metrics: total topics, exact duplicates, similar groups, unique topics.
   - Explore duplicate groups and recommendations for merging or rewording.
   - Visualize similarities with heatmaps and histograms.
   - Export grouped topics as a CSV for your editorial workflow.

## Example Usage

1. **Paste Topics** (e.g.):
    ```
    How to start a blog
    Blogging for beginners
    Starting your first blog
    Best blogging platforms
    WordPress vs other platforms
    ```
2. **Adjust Settings** (Sidebar):
   - Similarity threshold (strict vs. lenient matching)
   - Minimum group size

3. **Review Results**:
    - See grouped and unique topics.
    - Get actionable recommendations.

4. **Export**:
    - Download a CSV with all topics and their group assignments.

## Installation

1. **Clone this repository**
    ```bash
    git clone https://github.com/amal-alexander/topic-clustering.git
    cd topic-clustering
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app**
    ```bash
    streamlit run cluster.py
    ```

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- plotly

Install all dependencies using the provided `requirements.txt`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Optimize your blog content strategy with smart topic clustering!**

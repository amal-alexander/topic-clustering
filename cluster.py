import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import re
import io
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Blog Topic Duplicate Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .duplicate-group {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .cta-button {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        color: white;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        display: inline-block;
        text-decoration: none;
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_text(text):
    """Clean and preprocess text for better similarity detection"""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\-\']', '', text)
    # Remove common stop words and normalize similar terms
    text = text.replace('how to get', 'how to')
    text = text.replace('how to apply for', 'how to')
    text = text.replace('how to avail', 'how to')
    text = text.replace('via app', 'online')
    text = text.replace('through app', 'online')
    text = text.replace('instant', '')
    return text.strip()

def calculate_similarity_matrix(topics, method='tfidf'):
    """Calculate similarity matrix between topics"""
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Reduced from (1,3) for better matching
            stop_words='english',
            max_features=5000,   # Reduced features for better similarity
            min_df=1,            # Include all terms
            token_pattern=r'\b\w+\b'  # Better tokenization
        )
        tfidf_matrix = vectorizer.fit_transform(topics)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    
def find_duplicate_groups(topics, similarity_threshold=0.7):
    """Find groups of duplicate/similar topics"""
    preprocessed_topics = [preprocess_text(topic) for topic in topics]
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(preprocessed_topics)
    
    # Find exact duplicates first
    exact_duplicates = defaultdict(list)
    for i, topic in enumerate(preprocessed_topics):
        exact_duplicates[topic].append(i)
    
    # Remove single occurrences from exact duplicates
    exact_duplicates = {k: v for k, v in exact_duplicates.items() if len(v) > 1}
    
    # Find near duplicates using similarity threshold
    near_duplicates = defaultdict(list)
    used_indices = set()
    
    # Get indices of exact duplicates to exclude them
    for indices in exact_duplicates.values():
        used_indices.update(indices)
    
    # Group similar topics
    for i in range(len(topics)):
        if i in used_indices:
            continue
            
        group = [i]
        for j in range(i + 1, len(topics)):
            if j in used_indices:
                continue
                
            if similarity_matrix[i][j] >= similarity_threshold:
                group.append(j)
                used_indices.add(j)
        
        if len(group) > 1:
            near_duplicates[f"group_{len(near_duplicates)}"] = group
            used_indices.update(group)
    
    return exact_duplicates, near_duplicates, similarity_matrix

def generate_recommendations(duplicate_groups, topics):
    """Generate recommendations for handling duplicates"""
    recommendations = []
    
    for group_id, indices in duplicate_groups.items():
        group_topics = [topics[i] for i in indices]
        
        # Determine the best topic (usually the shortest or most comprehensive)
        best_topic = min(group_topics, key=len)
        
        rec = {
            'group_id': group_id,
            'topics': group_topics,
            'recommended_action': 'Merge or choose one',
            'suggested_topic': best_topic,
            'reason': f'Found {len(group_topics)} similar topics'
        }
        recommendations.append(rec)
    
    return recommendations

def create_similarity_heatmap(similarity_matrix, topics):
    """Create interactive similarity heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=[f"Topic {i+1}" for i in range(len(topics))],
        y=[f"Topic {i+1}" for i in range(len(topics))],
        colorscale='RdYlBu_r',
        text=[[f'{val:.2f}' for val in row] for row in similarity_matrix],
        texttemplate="%{text}",
        textfont={"size": 8},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Topic Similarity Matrix",
        xaxis_title="Topics",
        yaxis_title="Topics",
        height=600
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>🔍 Blog Topic Duplicate Detector</h1>
        <p>Identify and group duplicate or similar blog topics with AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Higher values = more strict matching (0.7 recommended for demos)"
    )
    
    min_group_size = st.sidebar.number_input(
        "Minimum Group Size",
        min_value=2,
        max_value=10,
        value=2,
        help="Minimum number of topics to form a group"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📝 Input Your Blog Topics</h3>
            <p>Paste your blog topics below (one per line) and let our AI identify duplicates and similar topics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Topic input
        topics_input = st.text_area(
            "Enter blog topics (one per line):",
            height=300,
            placeholder="How to start a blog\nBlogging for beginners\nStarting your first blog\nBest blogging platforms\nWordPress vs other platforms\n..."
        )
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Or upload a CSV file with topics",
            type=['csv'],
            help="CSV should have a column named 'topic' or 'title'"
        )
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>✨ Features</h3>
            <ul>
                <li>🎯 Exact duplicate detection</li>
                <li>🔍 Near-duplicate identification</li>
                <li>📊 Visual similarity analysis</li>
                <li>💡 Smart recommendations</li>
                <li>📥 CSV export functionality</li>
                <li>🎨 Interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Process topics
    topics = []
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'topic' in df.columns:
            topics = df['topic'].dropna().tolist()
        elif 'title' in df.columns:
            topics = df['title'].dropna().tolist()
        else:
            st.error("CSV file should have a 'topic' or 'title' column")
            return
    elif topics_input:
        topics = [topic.strip() for topic in topics_input.split('\n') if topic.strip()]
    
    if topics:
        if len(topics) < 2:
            st.warning("Please enter at least 2 topics to detect duplicates.")
            return
        
        with st.spinner("🔍 Analyzing topics for duplicates..."):
            exact_duplicates, near_duplicates, similarity_matrix = find_duplicate_groups(
                topics, similarity_threshold
            )
        
        # Display metrics
        st.markdown("## 📊 Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{len(topics)}</h3>
                <p>Total Topics</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            exact_dup_count = sum(len(indices) for indices in exact_duplicates.values())
            st.markdown(f"""
            <div class="metric-container">
                <h3>{exact_dup_count}</h3>
                <p>Exact Duplicates</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cluster_dup_count = sum(len(indices) for indices in near_duplicates.values())
            st.markdown(f"""
            <div class="metric-container">
                <h3>{cluster_dup_count}</h3>
                <p>Similar Topics</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            unique_topics = len(topics) - exact_dup_count - cluster_dup_count
            st.markdown(f"""
            <div class="metric-container">
                <h3>{unique_topics}</h3>
                <p>Unique Topics</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display duplicate groups
        if exact_duplicates or near_duplicates:
            st.markdown("## 🎯 Duplicate Groups")
            
            tab1, tab2, tab3 = st.tabs(["📋 Groups", "📊 Visualizations", "💡 Recommendations"])
            
            with tab1:
                # Exact duplicates
                if exact_duplicates:
                    st.markdown("### 🔍 Exact Duplicates")
                    for i, (key, indices) in enumerate(exact_duplicates.items()):
                        with st.expander(f"Group {i+1}: {len(indices)} identical topics"):
                            for idx in indices:
                                st.write(f"• {topics[idx]}")
                
                # Similar topics
                if near_duplicates:
                    st.markdown("### 🎯 Similar Topics")
                    for i, (group_id, indices) in enumerate(near_duplicates.items()):
                        # Calculate similarity scores for this group
                        similarities = []
                        for j in range(len(indices)):
                            for k in range(j+1, len(indices)):
                                sim_score = similarity_matrix[indices[j]][indices[k]]
                                similarities.append(sim_score)
                        avg_sim = np.mean(similarities) if similarities else 0
                        
                        with st.expander(f"Similarity Group {i+1}: {len(indices)} related topics (Avg similarity: {avg_sim:.2f})"):
                            for idx in indices:
                                st.write(f"• {topics[idx]}")
            
            with tab2:
                # Similarity heatmap
                if len(topics) <= 50:  # Only show for reasonable number of topics
                    fig = create_similarity_heatmap(similarity_matrix, topics)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Heatmap hidden for large datasets (>50 topics) for better performance")
                
                # Distribution chart
                similarity_scores = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
                fig_dist = px.histogram(
                    x=similarity_scores,
                    nbins=30,
                    title="Distribution of Topic Similarity Scores",
                    labels={"x": "Similarity Score", "y": "Frequency"}
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with tab3:
                # Generate recommendations
                all_groups = {}
                group_counter = 1
                
                for key, indices in exact_duplicates.items():
                    all_groups[f"Exact_{group_counter}"] = indices
                    group_counter += 1
                
                for group_id, indices in near_duplicates.items():
                    all_groups[f"Similar_{group_counter}"] = indices
                    group_counter += 1
                
                recommendations = generate_recommendations(all_groups, topics)
                
                for rec in recommendations:
                    st.markdown(f"""
                    <div class="duplicate-group">
                        <h4>🎯 {rec['group_id']}</h4>
                        <p><strong>Topics:</strong></p>
                        <ul>
                            {''.join([f"<li>{topic}</li>" for topic in rec['topics']])}
                        </ul>
                        <p><strong>Recommendation:</strong> {rec['recommended_action']}</p>
                        <p><strong>Suggested Topic:</strong> "{rec['suggested_topic']}"</p>
                        <p><strong>Reason:</strong> {rec['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Prepare data for CSV export
        export_data = []
        group_counter = 1
        
        # Add exact duplicates
        for key, indices in exact_duplicates.items():
            for idx in indices:
                export_data.append({
                    'Topic': topics[idx],
                    'Group_ID': f'Exact_Group_{group_counter}',
                    'Group_Type': 'Exact Duplicate',
                    'Group_Size': len(indices),
                    'Similarity_Score': 1.0
                })
            group_counter += 1
        
        # Add similar topics
        for group_id, indices in near_duplicates.items():
            # Calculate average similarity within the group
            group_similarities = []
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    group_similarities.append(similarity_matrix[indices[i]][indices[j]])
            avg_similarity = np.mean(group_similarities) if group_similarities else 0
            
            for idx in indices:
                export_data.append({
                    'Topic': topics[idx],
                    'Group_ID': f'Similar_Group_{group_counter}',
                    'Group_Type': 'Similar',
                    'Group_Size': len(indices),
                    'Similarity_Score': avg_similarity
                })
            group_counter += 1
        
        # Add unique topics
        used_indices = set()
        for indices in exact_duplicates.values():
            used_indices.update(indices)
        for indices in near_duplicates.values():
            used_indices.update(indices)
        
        for i, topic in enumerate(topics):
            if i not in used_indices:
                export_data.append({
                    'Topic': topic,
                    'Group_ID': 'Unique',
                    'Group_Type': 'Unique',
                    'Group_Size': 1,
                    'Similarity_Score': 0.0
                })
        
        # Create download button
        if export_data:
            df_export = pd.DataFrame(export_data)
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.markdown("## 📥 Export Results")
            st.download_button(
                label="📥 Download Grouped Topics (CSV)",
                data=csv_data,
                file_name="blog_topics_grouped.csv",
                mime="text/csv",
                help="Download all topics with their group assignments"
            )
            
            # Show preview
            with st.expander("📋 Preview Export Data"):
                st.dataframe(df_export.head(10))
        
        # Call-to-action
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h3>🚀 Ready to Optimize Your Content Strategy?</h3>
            <p>Use these insights to create a more focused and effective blog content plan!</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Show demo/instructions
        st.markdown("## 🎯 How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h4>1. 📝 Input Topics</h4>
                <p>Paste your blog topics or upload a CSV file with your content ideas.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h4>2. 🔍 AI Analysis</h4>
                <p>Our AI analyzes topics using advanced NLP to find duplicates and similarities.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-box">
                <h4>3. 📊 Get Results</h4>
                <p>View grouped topics, visualizations, and download organized results.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center;">
            <h3>✨ Start by entering your blog topics above!</h3>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
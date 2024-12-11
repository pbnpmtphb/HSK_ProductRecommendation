import streamlit as st
import pandas as pd
import pickle
from surprise import KNNWithMeans

## Data upload
df_collab = pd.read_csv('cleaned_output_collaborative.csv')
df_product = pd.read_csv('San_pham.csv')
df_comment = pd.read_csv('Danh_gia.csv')
df_product_details = pd.read_csv('products_with_ids.csv') 

## Open and read algorithms
with open('KNNWithMeans.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)

## Top 10 products
top_products = (
    df_comment['ma_san_pham']
    .value_counts()
    .head(10)
    .reset_index()
    .rename(columns={'index': 'Mã sản phẩm', 'ma_san_pham': 'Số lượt mua'})
)

top_products['Mã sản phẩm'] = top_products['Mã sản phẩm'].astype(str)
df_product_details['productid'] = df_product_details['productid'].astype(str)
df_product['ma_san_pham'] = df_product['ma_san_pham'].astype(str)
df_collab['ma_san_pham'] = df_collab['ma_san_pham'].astype(str)

df_merged = pd.merge(
    top_products,
    df_product_details,
    left_on='Mã sản phẩm',
    right_on='productid',
    how='left'
)
df_merged = pd.merge(
    df_merged,
    df_product,
    left_on='Mã sản phẩm',
    right_on='ma_san_pham',
    how='left'
)
df_result = df_merged[['Mã sản phẩm', 'name', 'Số lượt mua', 'diem_trung_binh', 'url']]

## Functions
def get_recommendation_collab(content, df_collab, df_product, df_product_details):
    df_rated = df_collab[(df_collab['ma_khach_hang'] == content) & (df_collab['so_sao'] >= 3)]
    rated_items = set(df_rated['ma_san_pham'])
    all_items = set(df_collab['ma_san_pham'].unique())
    unrated_items = all_items - rated_items
    df_score = pd.DataFrame(unrated_items, columns=['ma_san_pham'])
    df_score['EstimateScore'] = df_score['ma_san_pham'].apply(lambda x: knn.predict(content, x).est)
    df_score = df_score.sort_values(by='EstimateScore', ascending=False).reset_index(drop=True)
    df_recommendations = df_score[df_score['EstimateScore'] >= 3]
    result = df_recommendations.merge(df_product[["ma_san_pham", "ten_san_pham"]], on="ma_san_pham", how="left")
    result.rename(columns={"ma_san_pham": "ID sản phẩm", "ten_san_pham": "Tên sản phẩm", "EstimateScore": "Dự đoán đánh giá"}, inplace=True)
    result = result.head(10)
    result = pd.merge(
        result,
        df_product_details,
        left_on='ID sản phẩm',
        right_on='productid',
        how='left'
    )
    return result[['ID sản phẩm', 'Tên sản phẩm', 'Dự đoán đánh giá', 'url']]

def get_recommendation_content(product_id, cosine_sim, nums=10, min_rating=3.0):
    if product_id not in df_product['ma_san_pham'].values:
        return f"Product ID {product_id} not found in the dataset."
    idx = df_product[df_product['ma_san_pham'] == product_id].index[0]   
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    filtered_scores = [
        (i, score) for i, score in sim_scores[1:]  
        if df_product.iloc[i]['diem_trung_binh'] >= min_rating
    ]
    top_indices = [i[0] for i in filtered_scores[:nums]]
    result = df_product.iloc[top_indices][['ma_san_pham', 'ten_san_pham', 'diem_trung_binh']]
    result = pd.merge(
        result,
        df_product_details,
        left_on='ma_san_pham',
        right_on='productid',
        how='left'
    )
    return result[['ma_san_pham', 'ten_san_pham', 'diem_trung_binh', 'url']]

## CSS Styling
st.markdown("""
    <style>
    .sidebar-section {
        margin-bottom: 20px;
        padding: 10px;
        border-radius: 5px;
        background-color: #f5f5f5;
    }
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .custom-title {
        font-size: 30px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .custom-text {
        font-size: 18px;
        color: black;
        text-align: justify;
    }
    .error-label {
        font-size: 20px;
        font-weight: bold;
        color: red;
    }
    </style>
""", unsafe_allow_html=True)

## Streamlit Sidebar
st.sidebar.markdown("""
    <div class="sidebar-section">
        <p class="sidebar-title">Danh mục</p>
    </div>
""", unsafe_allow_html=True)
menu = ["Về dự án", "Tìm sản phẩm phù hợp"]
choice = st.sidebar.selectbox('', menu)
st.sidebar.markdown("""
    <div class="sidebar-section">
        <p class="sidebar-title">Thành viên thực hiện:</p>
        <p class="sidebar-info">Phạm Bích Nhật và Lương Nhã Hoàng Hà</p>
    </div>
    <div class="sidebar-section">
        <p class="sidebar-title">Giảng viên hướng dẫn:</p>
        <p class="sidebar-info">Cô Khuất Thùy Phương</p>
    </div>
    <div class="sidebar-section">
        <p class="sidebar-title">Thời gian thực hiện:</p>
        <p class="sidebar-info">12/2024</p>
    </div>
""", unsafe_allow_html=True)

## Main Content
if choice == 'Về dự án':  
    st.markdown("""
        <p class="custom-title">Mục tiêu của dự án</p>
        <p class="custom-text">Triển khai một Recommender System nhằm giúp khách hàng nhanh chóng tìm được sản phẩm phù hợp với nhu cầu cá nhân. Từ đó giúp Hasaki giữ chân khách hàng và tăng doanh số.</p>
        <hr style="border:1px solid #4CAF50;">
        <p class="custom-title">Giải pháp đề xuất</p>
        <p class="custom-text"><b>Collaborative Filtering:</b> Sử dụng dữ liệu đánh giá (số sao) từ cộng đồng người dùng để đề xuất sản phẩm dựa trên đánh giá tương đồng giữa các người dùng.</p>
        <p class="custom-text"><b>Content-Based Filtering:</b> Phân tích đặc điểm sản phẩm được tương tác bởi người dùng để đề xuất những sản phẩm tương tự.</p>
    """, unsafe_allow_html=True)

elif choice == "Tìm sản phẩm phù hợp":
    st.markdown("""
        <p class="custom-title">Nhập ID để đăng nhập:</p>
    """, unsafe_allow_html=True)

    user_input = st.text_input("", "")

    if user_input:
        if user_input not in df_collab['ma_khach_hang'].astype(str).values:
            st.markdown("""
            <p class="error-label">Mã ID không tồn tại!</p>
            """, unsafe_allow_html=True)
            st.dataframe(df_result, hide_index=True)
        else:
            recommendations = get_recommendation_collab(user_input, df_collab, df_product, df_product_details)
            st.markdown("""
            <p class="custom-text">Bạn có hứng thú với những sản phẩm bên dưới không?</p>
            """, unsafe_allow_html=True)
            st.dataframe(recommendations, hide_index=True)

            selected_product = st.selectbox("Chọn sản phẩm:", df_product['ten_san_pham'].unique())
            if selected_product:
                product_id = df_product[df_product['ten_san_pham'] == selected_product]['ma_san_pham'].values[0]
                recommendations_content = get_recommendation_content(product_id, cosine_sim_new)
                st.dataframe(recommendations_content, hide_index=True)
import streamlit as st
import pandas as pd
import pickle
import requests
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

## Top 9 products
top_products = (
    df_comment['ma_san_pham']
    .value_counts()
    .head(9)
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
df_result = df_merged[['Mã sản phẩm', 'name', 'Số lượt mua', 'diem_trung_binh','url']]

## Functions
def get_recommendation_collab(content, df_collab, df_product, df_product_details): ## content == userid  
    # Filter products the user has already rated with rating no lower than 3
    df_rated = df_collab[(df_collab['ma_khach_hang'] == content) & (df_collab['so_sao'] >=3)]
    rated_items = set(df_rated['ma_san_pham'])
    # Create a DataFrame of all products not rated by the user
    all_items = set(df_collab['ma_san_pham'].unique())
    unrated_items = all_items - rated_items
    df_score = pd.DataFrame(unrated_items, columns=['ma_san_pham'])
    # Predict scores for unrated items
    df_score['EstimateScore'] = df_score['ma_san_pham'].apply(lambda x: knn.predict(content, x).est)
    # Sort by estimated score in descending order
    df_score = df_score.sort_values(by='EstimateScore', ascending=False).reset_index(drop=True)
    df_recommendations = df_score[df_score['EstimateScore'] >= 3]
    result = df_recommendations.merge(df_product[["ma_san_pham", "ten_san_pham"]], on="ma_san_pham", how="left")
    result.rename(columns={"ma_san_pham": "ID sản phẩm", "ten_san_pham": "Tên sản phẩm", "EstimateScore": "Dự đoán đánh giá"}, inplace=True)
    result = result[["ID sản phẩm", "Tên sản phẩm", "Dự đoán đánh giá"]]
    result = result.head(9)
    # Merge
    result = pd.merge(
        result,
        df_product_details,
        left_on='ID sản phẩm',
        right_on='productid',
        how='left'
    )
    result["Dự đoán đánh giá"] = result["Dự đoán đánh giá"].round(1)
    return result[['ID sản phẩm', 'Tên sản phẩm', 'Dự đoán đánh giá', 'url']]

def get_recommendation_content(product_id, cosine_sim, nums=9, min_rating=3.0):
    if product_id not in df_product['ma_san_pham'].values:
        return f"Product ID {product_id} not found in the dataset."
    idx = df_product[df_product['ma_san_pham'] == product_id].index[0]   
    # Calculate similarity scores for the product
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Exclude the input product itself and apply the rating filter
    filtered_scores = [
        (i, score) for i, score in sim_scores[1:]  # Skip the first (itself)
        if df_product.iloc[i]['diem_trung_binh'] >= min_rating
    ]
    # Select top 'nums' similar products
    top_indices = [i[0] for i in filtered_scores[:nums]]
    # Return the top similar products as a DataFrame
    result = df_product.iloc[top_indices][['ma_san_pham', 'ten_san_pham', 'diem_trung_binh']]
    result = pd.merge(
        result,
        df_product_details,
        left_on='ma_san_pham',
        right_on='productid',
        how='left'
    )
    return result[['ma_san_pham', 'ten_san_pham', 'diem_trung_binh', 'url']]


## Streamlit interface
st.image("hasaki_banner.jpg", use_column_width=True)

menu = ["Về dự án", "Tìm sản phẩm phù hợp"]
choice = st.sidebar.selectbox('Danh mục', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Phạm Bích Nhật và Lương Nhã Hoàng Hà""")
st.sidebar.write("""#### Giảng viên hướng dẫn: 
                 Cô Khuất Thùy Phương """)
st.sidebar.write("""#### Thời gian thực hiện: 
                 12/2024""")

if choice == 'Về dự án':  
    st.markdown("""
        <style>
        .custom-label-1 {
            font-size: 30px;
            font-weight: bold;
            color: #4CAF50;
            font-family: 'Arial', sans-serif;
        }
        .info-label-1 {
            font-size: 20px;
            font-weight: normal;
            color: black;
            font-family: 'Arial', sans-serif;
        }
        </style>
        <p class="custom-label-1">Mục tiêu của dự án</p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p class="info-label-1">Triển khai một Recommender System nhằm giúp khách hàng nhanh chóng tìm được sản phẩm phù hợp với nhu cầu cá nhân. Từ đó giúp Hasaki giữ chân khách hàng và tăng doanh số.</p>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <p class="custom-label-1">Giải pháp đề xuất</p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p class="info-label-1"><b>Collaborative Filtering:</b> Sử dụng dữ liệu đánh giá (số sao) từ cộng đồng người dùng để đề xuất sản phẩm dựa trên đánh giá tương đồng giữa các người dùng.</p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p class="info-label-1"><b>Content-Based Filtering:</b> Phân tích đặc điểm sản phẩm được tương tác bởi người dùng để đề xuất những sản phẩm tương tự.</p>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <p class="custom-label-1">Lợi ích nổi bật của giải pháp</p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p class="info-label-1"><b>Tăng trải nghiệm khách hàng:</b> Gợi ý chính xác sản phẩm mà khách hàng quan tâm, giúp họ tiết kiệm thời gian và nâng cao sự hài lòng.</p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p class="info-label-1"><b>Tăng doanh số:</b> Đề xuất đúng sản phẩm cho đúng khách hàng sẽ kích thích hành vi mua sắm.</p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p class="info-label-1"><b>Cá nhân hóa:</b> Hệ thống học hỏi và tùy chỉnh theo từng khách hàng, tạo sự khác biệt so với đối thủ.</p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p class="info-label-1"><b>Giữ chân khách hàng:</b> Gắn kết người dùng lâu dài với nền tảng thông qua trải nghiệm mua sắm thông minh.</p>
    """, unsafe_allow_html=True)

elif choice == "Tìm sản phẩm phù hợp":
    st.markdown("""
        <style>
        .custom-label {
            font-size: 30px;
            font-weight: bold;
            color: #4CAF50;
            font-family: 'Arial', sans-serif;
        }
        .error-label {
            font-size: 30px;
            font-weight: bold;
            color: red;
            font-family: 'Arial', sans-serif;
        }
        .info-label {
            font-size: 30px;
            font-weight: bold;
            color: black;
            font-family: 'Arial', sans-serif;
        }
        </style>
        <p class="custom-label">Nhập ID để đăng nhập:</p>
    """, unsafe_allow_html=True)
    
    content = st.text_area(label="", height=68)

    if content != "":
        if content not in df_collab['ma_khach_hang'].astype(str).values:
            # Error message for invalid ID
            st.markdown("""
            <p class="error-label">Mã ID không tồn tại!</p>
            """, unsafe_allow_html=True)
            
            # Additional black message
            st.markdown("""
            <p class="info-label">Hãy tạo tài khoản để mua những sản phẩm bán chạy nhất của Hasaki nhé!</p>
            """, unsafe_allow_html=True)
            row1 = st.columns(3)
            row2 = st.columns(3)
            row3 = st.columns(3)
            rows = [row1, row2, row3]
            for i, col in enumerate(row1 + row2 + row3):
                if i < len(df_result):
                    product = df_result.iloc[i]
                    with col:
                        st.subheader(product['name'])  
                        st.markdown(f"**ID sản phẩm:** {product['Mã sản phẩm']}")  
                        st.markdown(f"**Số lượt mua:** {product['Số lượt mua']}")
                        st.markdown(f"**Điểm trung bình:** {product['diem_trung_binh']} ⭐")  
                        st.markdown(f"[Xem sản phẩm]({product['url']})")
            st.divider()

            # Content base
            st.markdown(f"""
            <p class="info-label">Hãy chọn sản phẩm trong danh sách dưới đây để tham khảo nhanh nhé!</p>
            """, unsafe_allow_html=True)
            product_names = df_product['ten_san_pham'].unique()
            selected_product = st.selectbox("Chọn sản phẩm:", product_names)
            product_description = df_product.loc[df_product['ten_san_pham'] == selected_product, 'mo_ta'].values
            if len(product_description) > 0:
                product_description = df_product.loc[df_product['ten_san_pham'] == selected_product, 'mo_ta'].values[0]
                product_url = df_product_details.loc[df_product_details['productid'] == df_product.loc[df_product['ten_san_pham'] == selected_product, 'ma_san_pham'].values[0], 'url'].values[0]
                product_id = df_product.loc[df_product['ten_san_pham'] == selected_product, 'ma_san_pham'].values[0]

                st.write(f"Sản phẩm bạn đã chọn là: {selected_product}")
                st.markdown(f"[Xem sản phẩm chi tiết tại đây]({product_url})")
                with st.expander("Mô tả sản phẩm", expanded=False):
                    st.write(product_description)
            else:
                st.write("Thông tin sản phẩm không khả dụng.")
            
            st.markdown(f"""
            <p class="info-label">Cùng xem một số sản phẩm tương tự sản phẩm bạn đã chọn nhé!</p>
            """, unsafe_allow_html=True)
            recommendations_1 = get_recommendation_content(product_id, cosine_sim=cosine_sim_new, nums=10, min_rating=3.0)
            row10 = st.columns(3)
            row11 = st.columns(3)
            row12 = st.columns(3)
            rows = [row10, row11, row12]
            for i, col in enumerate(row10 + row11 + row12):
                if i < len(recommendations_1):
                    product2 = recommendations_1.iloc[i]
                    with col:
                        st.subheader(product2['ten_san_pham'])  
                        st.markdown(f"**ID sản phẩm:** {product2['ma_san_pham']}")  
                        st.markdown(f"**Điểm trung bình:** {product2['diem_trung_binh']} ⭐")  
                        st.markdown(f"[Xem sản phẩm]({product2['url']})")  

        else:
            st.markdown(f"""
            <p class="custom-label">Xin chào khách hàng với ID: <strong>{content}</strong>!</p>
            """, unsafe_allow_html=True)  
            recommendations = get_recommendation_collab(content, df_collab, df_product, df_product_details)
            st.markdown("""
            <p class="info-label">Bạn có hứng thú với những sản phẩm bên dưới không?</p>
            """, unsafe_allow_html=True)
            row4 = st.columns(3)
            row5 = st.columns(3)
            row6 = st.columns(3)
            rows = [row4, row5, row6]
            for i, col in enumerate(row4 + row5 + row6):
                if i < len(recommendations):
                    product1 = recommendations.iloc[i]
                    with col:
                        st.subheader(product1['Tên sản phẩm'])  
                        st.markdown(f"**ID sản phẩm:** {product1['ID sản phẩm']}")  
                        st.markdown(f"**Dự đoán đánh giá:** {product1['Dự đoán đánh giá']} ⭐")  
                        st.markdown(f"[Xem sản phẩm]({product1['url']})")  
            st.divider()

            st.markdown(f"""
            <p class="info-label">Vẫn chưa tìm thấy sản phẩm bạn thích? Hãy chọn sản phẩm trong danh sách dưới đây!</p>
            """, unsafe_allow_html=True)
            product_names = df_product['ten_san_pham'].unique()
            selected_product = st.selectbox("Chọn sản phẩm:", product_names)
            product_description = df_product.loc[df_product['ten_san_pham'] == selected_product, 'mo_ta'].values
            if len(product_description) > 0:
                product_description = df_product.loc[df_product['ten_san_pham'] == selected_product, 'mo_ta'].values[0]
                product_url = df_product_details.loc[df_product_details['productid'] == df_product.loc[df_product['ten_san_pham'] == selected_product, 'ma_san_pham'].values[0], 'url'].values[0]
                product_id = df_product.loc[df_product['ten_san_pham'] == selected_product, 'ma_san_pham'].values[0]

                st.write(f"Sản phẩm bạn đã chọn là: {selected_product}")
                st.markdown(f"[Xem sản phẩm chi tiết tại đây]({product_url})")
                with st.expander("Mô tả sản phẩm", expanded=False):
                    st.write(product_description)
            else:
                st.write("Thông tin sản phẩm không khả dụng.")
            
            st.markdown(f"""
            <p class="info-label">Cùng xem một số sản phẩm tương tự sản phẩm bạn đã chọn nhé!</p>
            """, unsafe_allow_html=True)
            recommendations_1 = get_recommendation_content(product_id, cosine_sim=cosine_sim_new, nums=10, min_rating=3.0)
            row7 = st.columns(3)
            row8 = st.columns(3)
            row9 = st.columns(3)
            rows = [row7, row8, row9]
            for i, col in enumerate(row7 + row8 + row9):
                if i < len(recommendations_1):
                    product2 = recommendations_1.iloc[i]
                    with col:
                        st.subheader(product2['ten_san_pham'])  
                        st.markdown(f"**ID sản phẩm:** {product2['ma_san_pham']}")  
                        st.markdown(f"**Điểm trung bình:** {product2['diem_trung_binh']} ⭐")  
                        st.markdown(f"[Xem sản phẩm]({product2['url']})")  

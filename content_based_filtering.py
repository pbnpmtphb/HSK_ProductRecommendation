import streamlit as st
import pandas as pd
import pickle

# function cần thiết

def get_recommendation(product_id, cosine_sim, nums=5, min_rating=3.0):

    if product_id not in df_products['ma_san_pham'].values:
        return f"Product ID {product_id} not found in the dataset."
    
    # Find the index of the product with the given product_id
    idx = df_products[df_products['ma_san_pham'] == product_id].index[0]   

    # Calculate similarity scores for the product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the input product itself and apply the rating filter
    filtered_scores = [
        (i, score) for i, score in sim_scores[1:]  # Skip the first (itself)
        if df_products.iloc[i]['diem_trung_binh'] >= min_rating
    ]

    # Select top 'nums' similar products
    top_indices = [i[0] for i in filtered_scores[:nums]]

    # Return the top similar products as a DataFrame
    return df_products.iloc[top_indices][['ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh']]

def display_recommended_products(recommended_products, cols=5):
    for i in range(0, len(recommended_products), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:   
                    st.write(product['ten_san_pham'])                    
                    expander = st.expander(f"Mô tả")
                    product_description = product['mo_ta']
                    truncated_description = ' '.join(product_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           

# Đọc dữ liệu sản phẩm
df_products = pd.read_csv('San_pham.csv')
# Lấy 10 sản phẩm
random_products = df_products.head(n=10)
# print(random_products)

st.session_state.df_products = df_products

# Open and read file to cosine_sim_new
with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)

###### Giao diện Streamlit ######
menu = ["Về dự án", "Tìm sản phẩm"]  
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Phạm Bích Nhật và Lương Nhã Hoàng Hà""")
st.sidebar.write("""#### Giảng viên hướng dẫn: 
                 Cô Khuất Thùy Phương """)
st.sidebar.write("""#### Thời gian thực hiện: 
                 12/2024""")
st.image('hasaki_banner.jpg', use_column_width=True)

if choice == 'Về dự án':  
    # Trang trí trong trang mở đầu
    st.markdown("*Our recommendation system* help you pick the **right** ***products***.")
    st.markdown('''
        :red[Hasaki] :orange[can] :green[offer] :blue[you] :violet[the]
        :gray[most] :rainbow[relevant] and :blue-background[high-rated] products for your wellbeing .''')

elif choice == "Tìm sản phẩm":
    # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
    if 'selected_ma_san_pham' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
        st.session_state.selected_ma_san_pham = None

    # Theo cách cho người dùng chọn sản phẩm từ dropdown
    # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in st.session_state.df_products.iterrows()]
    st.session_state.df_products
    # Tạo một dropdown với options là các tuple này
    selected_product = st.selectbox(
        "Chọn sản phẩm",
        options=product_options,
        format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
    )
    # Display the selected product
    st.write("Bạn đã chọn:", selected_product)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_ma_san_pham = selected_product[1]

    if st.session_state.selected_ma_san_pham:
        st.write("ma_san_pham: ", st.session_state.selected_ma_san_pham)
        # Hiển thị thông tin sản phẩm được chọn
        selected_product = df_products[df_products['ma_san_pham'] == st.session_state.selected_ma_san_pham]

        if not selected_product.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_product['ten_san_pham'].values[0])

            product_description = selected_product['mo_ta'].values[0]
            truncated_description = ' '.join(product_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')

            st.write('##### Các sản phẩm liên quan:')
            recommendations = get_recommendation(st.session_state.selected_ma_san_pham, cosine_sim=cosine_sim_new, nums=5, min_rating=3.0) 
            display_recommended_products(recommendations, cols=5)
        else:
            st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")
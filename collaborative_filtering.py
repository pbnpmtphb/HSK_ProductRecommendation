import streamlit as st
import pandas as pd
import pickle
from surprise import KNNWithMeans

# function cần thiết
def get_recommendation(content, df_new, df_product): ## content == userid  
# Filter products the user has already rated with rating no lower than 3
    df_rated = df_new[(df_new['ma_khach_hang'] == content) & (df_new['so_sao'] >=3)]
    rated_items = set(df_rated['ma_san_pham'])

    # Create a DataFrame of all products not rated by the user
    all_items = set(df_new['ma_san_pham'].unique())
    unrated_items = all_items - rated_items
    df_score = pd.DataFrame(unrated_items, columns=['ma_san_pham'])

    # Predict scores for unrated items
    df_score['EstimateScore'] = df_score['ma_san_pham'].apply(lambda x: knn.predict(content, x).est)

    # Sort by estimated score in descending order
    df_score = df_score.sort_values(by='EstimateScore', ascending=False).reset_index(drop=True)
    df_recommendations = df_score[df_score['EstimateScore'] >= 3]

    result = df_recommendations.merge(df_product[["ma_san_pham", "ten_san_pham"]], on="ma_san_pham", how="left")

    result.rename(columns={"ma_san_pham": "Mã sản phẩm", "ten_san_pham": "Tên sản phẩm", "EstimateScore": "rating"}, inplace=True)
    result = result[["Mã sản phẩm", "Tên sản phẩm"]]
    return result.head(10)

# Đọc dữ liệu sản phẩm và khách hàng
df_new = pd.read_csv('cleaned_output_collaborative.csv')
df_product = pd.read_csv('San_pham.csv')

# Open and read algorithm
with open('KNNWithMeans.pkl', 'rb') as f:
    knn = pickle.load(f)

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
    content = st.text_area(label="Nhập ID để đăng nhập:")
    if content!="":
        if content not in df_new['ma_khach_hang'].astype(str).values:
            st.write("ID này không tồn tại.")
        else:
            st.write("Xin chào khách hàng ", content, "!")
            recommendations = get_recommendation(content, df_new, df_product)
            st.write("Bạn có hứng thú với những sản phẩm bên dưới không?")  
            pd.set_option('display.max_colwidth', None)      
            st.code(recommendations)
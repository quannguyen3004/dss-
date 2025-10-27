# TOPSIS_Dashboard_Full.py
# Ứng dụng tích hợp Kết nối DB, Import CSV và Dashboard AHP-TOPSIS Streamlit

# --- PHẦN 1: IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf # Thư viện để lấy dữ liệu chứng khoán
import plotly.express as px
import itertools # Để tạo các cặp so sánh cho AHP

# Thư viện cho MySQL và Import CSV
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine 

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="TOPSIS - Phân tích Rủi ro Cổ phiếu",
    page_icon="⚖️",
    layout="wide"
)

# --- 2. CÁC HÀM XỬ LÝ LÕI (AHP VÀ TOPSIS) ---

@st.cache_data
def get_stock_data(tickers_list):
    """
    Hàm lấy các chỉ số tài chính quan trọng cho việc phân tích rủi ro từ Yahoo Finance.
    """
    financial_data = []
    for ticker_symbol in tickers_list:
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # Lấy các chỉ số
            data = {
                'Mã CP': ticker_symbol,
                'Beta': info.get('beta'),
                'P/E': info.get('trailingPE'),
                'Nợ/Vốn CSH': info.get('debtToEquity'),
                'ROE': info.get('returnOnEquity'),
                'Biên LN': info.get('profitMargins')
            }
            financial_data.append(data)
        except Exception as e:
            # st.warning(f"Không thể lấy dữ liệu cho mã '{ticker_symbol}'. Vui lòng kiểm tra lại mã cổ phiếu.")
            pass # Bỏ qua lỗi và tiếp tục với các mã khác
            
    df = pd.DataFrame(financial_data).set_index('Mã CP')
    
    # Xử lý missing values bằng giá trị trung vị của cột
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    return df

def calculate_ahp_weights(comparison_matrix):
    """
    Hàm tính trọng số (eigenvector) của ma trận so sánh cặp AHP và CR.
    """
    try:
        A = np.array(comparison_matrix)
        n = A.shape[0]

        # Tính giá trị riêng và vector riêng
        eigenvalues, eigenvectors = np.linalg.eig(A)
        max_eigen_value = np.max(eigenvalues.real)
        
        max_eigen_vector_index = np.argmax(eigenvalues.real)
        weights = eigenvectors[:, max_eigen_vector_index].real
        weights = weights / weights.sum() # Chuẩn hóa trọng số
        
        # Tính Chỉ số nhất quán (CI)
        CI = (max_eigen_value - n) / (n - 1)
        
        # Chỉ số ngẫu nhiên (RI)
        RI_values = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        RI = RI_values.get(n, 1.49) 

        # Tỷ lệ nhất quán (CR)
        CR = CI / RI if RI != 0 else 0
        
        return weights, CR
    except Exception as e:
        return np.array([1/n for _ in range(n)]), 1.0 

def run_topsis(decision_matrix, weights, impacts):
    """
    Hàm thực thi thuật toán TOPSIS để xếp hạng các lựa chọn.
    """
    # Bước 1: Chuẩn hóa ma trận quyết định
    norm_matrix = decision_matrix / np.sqrt((decision_matrix**2).sum(axis=0))
    
    # Bước 2: Tính toán ma trận đã được chuẩn hóa và có trọng số
    weighted_matrix = norm_matrix * weights
    
    # Bước 3: Xác định giải pháp lý tưởng tốt nhất (A+) và tệ nhất (A-)
    ideal_best = np.zeros(weighted_matrix.shape[1])
    ideal_worst = np.zeros(weighted_matrix.shape[1])
    
    for i in range(len(impacts)):
        if impacts[i] == 1: # Benefit (càng cao càng tốt)
            ideal_best[i] = weighted_matrix.iloc[:, i].max()
            ideal_worst[i] = weighted_matrix.iloc[:, i].min()
        else: # Cost (càng thấp càng tốt)
            ideal_best[i] = weighted_matrix.iloc[:, i].min()
            ideal_worst[i] = weighted_matrix.iloc[:, i].max()
            
    # Bước 4: Tính khoảng cách Euclidean đến giải pháp lý tưởng
    dist_to_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_to_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    
    # Bước 5: Tính điểm TOPSIS
    epsilon = 1e-6
    topsis_score = dist_to_worst / (dist_to_best + dist_to_worst + epsilon)
    
    return topsis_score


# --- 3. PHẦN XỬ LÝ KẾT NỐI VÀ IMPORT CSV (Chỉ chạy một lần khi khởi động) ---

# Cấu hình kết nối
db_host = '127.0.0.1' 
db_user = 'root'
db_password = '130225' 
db_name = 'tickets'    
db_port = 3306         
connection = None  
csv_file_path = 'du_lieu_mau_co_phieu.csv' # <== THAY THẾ bằng tên file CSV
table_name = 'du_lieu_topsis'              # Tên bảng đích

# Streamlit cache để đảm bảo code này chỉ chạy một lần
@st.cache_resource
def run_db_operations(db_host, db_user, db_password, db_name, db_port, csv_file_path, table_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=db_port
        )

        if connection.is_connected():
            st.toast("✅ Kết nối MySQL thành công!", icon="💾")

            # --- LOGIC IMPORT CSV ---
            try:
                df_import = pd.read_csv(csv_file_path)
                
                # Cần cài: pip install sqlalchemy mysql-connector-python
                db_url = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                engine = create_engine(db_url)
                
                df_import.to_sql(
                    name=table_name, 
                    con=engine, 
                    if_exists='replace',
                    index=False
                )
                st.toast(f"✅ Đã import CSV vào bảng '{table_name}' thành công!", icon="🚀")
                
            except FileNotFoundError:
                st.warning(f"❌ Cảnh báo: Không tìm thấy file CSV '{csv_file_path}'. Bỏ qua Import.")
            except ImportError:
                 st.error("❌ Lỗi Thư viện: Vui lòng cài đặt 'sqlalchemy' để sử dụng chức năng Import CSV.")
            except Exception as e:
                st.error(f"❌ Lỗi khi import CSV: {e}")

            # --- TRUY VẤN KIỂM TRA BẢNG MẪU ---
            cursor = connection.cursor()
            query = f"SELECT COUNT(*) FROM {table_name};"
            cursor.execute(query)
            count = cursor.fetchone()[0]
            st.sidebar.markdown(f"**Trạng thái DB:** Bảng `{table_name}` có **{count}** dòng.")
            cursor.close()

    except Error as e:
        st.error(f"❌ Lỗi Kết nối MySQL: {e}. Vui lòng kiểm tra Server và thông tin đăng nhập.")
    finally:
        if connection and connection.is_connected():
            connection.close()

# Thực thi các thao tác DB khi ứng dụng khởi động lần đầu
run_db_operations(db_host, db_user, db_password, db_name, db_port, csv_file_path, table_name)


# --- 4. XÂY DỰNG GIAO DIỆN NGƯỜI DÙNG (STREAMLIT UI) ---

st.title("⚖️ Ứng dụng Phân tích Rủi ro Đầu tư Cổ phiếu bằng TOPSIS & AHP")
st.info("Ứng dụng sử dụng **AHP** để xác định trọng số tiêu chí và **TOPSIS** để xếp hạng rủi ro của cổ phiếu (dữ liệu lấy từ Yahoo Finance).")

# Định nghĩa các tiêu chí và tác động của chúng
criteria = {
    'Beta': -1, 'P/E': -1, 'Nợ/Vốn CSH': -1,
    'ROE': 1, 'Biên LN': 1
}
criteria_list = list(criteria.keys())
n_criteria = len(criteria_list)


# --- Sidebar: Nơi người dùng nhập liệu và cấu hình ---
with st.sidebar:
    st.header("⚙️ Bảng điều khiển & Dữ liệu")
    
    # Nhập danh sách cổ phiếu
    tickers_input = st.text_area(
        "Nhập các mã cổ phiếu (cách nhau bởi dấu phẩy hoặc xuống dòng):",
        "AAPL, MSFT, GOOGL, TSLA, NVDA"
    )
    
    st.header("⚖️ Thiết lập Trọng số bằng AHP")
    st.markdown("Sử dụng thanh trượt để so sánh độ quan trọng của các tiêu chí (1/9...9).")
    
    # Khởi tạo ma trận so sánh
    comparison_values = {}
    
    # 1. Tạo giao diện so sánh cặp
    for crit_i, crit_j in itertools.combinations(criteria_list, 2):
        key = tuple(sorted((crit_i, crit_j)))
        
        comparison_value = st.select_slider(
            f"**{crit_i}** quan trọng hơn **{crit_j}** bao nhiêu?",
            options=[1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            value=1.0,
            format_func=lambda x: f"1/{round(1/x)}" if x < 1 else str(round(x))
        )
        comparison_values[key] = comparison_value
        

    # 2. Xây dựng ma trận so sánh dựa trên input
    comparison_matrix = [[0.0] * n_criteria for _ in range(n_criteria)]
    
    for i in range(n_criteria):
        for j in range(n_criteria):
            if i == j:
                comparison_matrix[i][j] = 1.0 # Đường chéo chính
            elif i < j:
                crit_i = criteria_list[i]
                crit_j = criteria_list[j]
                key = tuple(sorted((crit_i, crit_j)))
                comparison_matrix[i][j] = comparison_values[key]
            else:
                comparison_matrix[i][j] = 1.0 / comparison_matrix[j][i]
                
    # 3. Tính toán trọng số AHP và CR
    ahp_weights_array, CR = calculate_ahp_weights(comparison_matrix)
    
    st.subheader("Kết quả AHP:")
    st.markdown(f"**Tỷ lệ nhất quán (CR):** `{CR:.4f}`")
    
    # Tạo dictionary trọng số (chỉ khi CR thấp)
    if CR > 0.1:
        st.error("⚠️ **CR > 0.1**. Mức độ nhất quán thấp. Vui lòng điều chỉnh các so sánh cặp.")
        weights = {}
    else:
        st.success("✅ **CR < 0.1**. Mức độ nhất quán chấp nhận được.")
        
        weights = {criteria_list[i]: ahp_weights_array[i] for i in range(n_criteria)}
        
        st.markdown("**Trọng số được tính toán:**")
        weights_df = pd.DataFrame(weights.items(), columns=['Tiêu chí', 'Trọng số']).set_index('Tiêu chí')
        st.dataframe(weights_df.style.format("{:.4f}"))


# --- Nút bắt đầu phân tích TOPSIS ---
if st.button("🚀 Bắt đầu Phân tích TOPSIS", use_container_width=True):
    # Xử lý input của người dùng
    tickers_list = [t.strip().upper() for t in tickers_input.replace(',', '\n').split('\n') if t.strip()]
    
    if not tickers_list:
        st.error("Vui lòng nhập ít nhất một mã cổ phiếu.")
    elif CR > 0.1:
        st.warning("Vui lòng sửa lỗi Mức độ nhất quán (CR > 0.1) trong bảng điều khiển AHP trước khi chạy TOPSIS.")
    elif not weights:
        st.warning("Không thể tính trọng số AHP hợp lệ. Vui lòng kiểm tra lại cấu hình.")
    else:
        # --- 5. LẤY VÀ HIỂN THỊ DỮ LIỆU ---
        with st.spinner(f"Đang tải dữ liệu cho {len(tickers_list)} cổ phiếu..."):
            raw_data = get_stock_data(tickers_list)
        
        if not raw_data.empty:
            st.header("📊 Dữ liệu Tài chính thô")
            st.dataframe(raw_data.style.format("{:.4f}"))
            
            # --- 6. THỰC THI TOPSIS VÀ HIỂN THỊ KẾT QUẢ ---
            st.header("🏆 Kết quả Xếp hạng Rủi ro")
            
            # Chuẩn bị dữ liệu cho hàm TOPSIS
            decision_matrix = raw_data[list(criteria.keys())]
            weights_list = np.array([weights[crit] for crit in criteria])
            impacts_list = np.array([criteria[crit] for crit in criteria])
            
            # Chạy thuật toán
            scores = run_topsis(decision_matrix, weights_list, impacts_list)
            
            # Tạo DataFrame kết quả
            results_df = raw_data.copy()
            results_df['Điểm TOPSIS'] = scores
            results_df['Xếp hạng'] = results_df['Điểm TOPSIS'].rank(ascending=False).astype(int)
            results_df = results_df.sort_values(by='Xếp hạng')
            
            st.success("**Diễn giải:** Cổ phiếu có **Xếp hạng 1** và **Điểm TOPSIS cao nhất** được đánh giá là lựa chọn **ít rủi ro nhất**.")
            st.dataframe(
                results_df.style.format("{:.4f}").background_gradient(
                    cmap='Greens', subset=['Điểm TOPSIS']
                ), 
                use_container_width=True
            )
            
            # --- 7. TRỰC QUAN HÓA KẾT QUẢ ---
            st.header("🎨 Trực quan hóa Kết quả")
            fig = px.bar(
                results_df,
                x=results_df.index,
                y='Điểm TOPSIS',
                color='Điểm TOPSIS',
                color_continuous_scale='Greens',
                title='So sánh Điểm TOPSIS giữa các Cổ phiếu',
                labels={'Mã CP': 'Mã Cổ phiếu', 'Điểm TOPSIS': 'Điểm TOPSIS (Càng cao càng tốt)'}
            )
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)
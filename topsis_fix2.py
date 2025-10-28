import mysql.connector
from mysql.connector import Error
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import itertools # Thêm thư viện để tạo các cặp so sánh cho AHP

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="TOPSIS - Phân tích Rủi ro Cổ phiếu",
    page_icon="⚖️",
    layout="wide"
)

# --- CÁC BIẾN CỐ ĐỊNH CHO LUỒNG DATABASE ---
DB_CRITERIA = ['RSI', 'MACD', 'trailingPE_snapshot', 'marketCap_snapshot', 'Returns']
DB_IMPACTS = {
    'RSI': 'Cost',
    'MACD': 'Benefit',
    'trailingPE_snapshot': 'Cost',
    'marketCap_snapshot': 'Benefit',
    'Returns': 'Benefit'
}

# === PHẦN MỚI: Thêm dictionary giải thích các chỉ số ===
CRITERIA_EXPLANATIONS = {
    # Yahoo Finance
    'Beta': "Đo lường mức độ biến động của cổ phiếu so với thị trường chung. Beta < 1 cho thấy ít biến động hơn thị trường (ít rủi ro hơn).",
    'P/E': "Tỷ lệ Giá trên Thu nhập. Chỉ số này cho biết nhà đầu tư sẵn sàng trả bao nhiêu cho một đồng lợi nhuận. P/E quá cao có thể là dấu hiệu định giá đắt.",
    'Nợ/Vốn CSH': "Tỷ lệ Nợ trên Vốn chủ sở hữu, đo lường đòn bẩy tài chính. Tỷ lệ < 1 thường được coi là an toàn.",
    'ROE': "Lợi nhuận trên Vốn chủ sở hữu. Đo lường khả năng sinh lời của công ty. ROE > 15% thường được xem là tốt.",
    'Biên LN': "Biên Lợi nhuận. Cho biết công ty tạo ra bao nhiêu lợi nhuận từ doanh thu. Biên lợi nhuận càng cao càng tốt.",
    # Database
    'RSI': "Chỉ số Sức mạnh Tương đối. Đo lường tốc độ và sự thay đổi của các biến động giá. RSI > 70 cho thấy tín hiệu 'quá mua' (có thể sớm điều chỉnh giảm).",
    'MACD': "Đường Trung bình động hội tụ/phân kỳ. Là một chỉ báo xu hướng. MACD > 0 thường báo hiệu xu hướng tăng.",
    'trailingPE_snapshot': "Tương tự P/E, tỷ lệ giá trên thu nhập trong 12 tháng gần nhất.",
    'marketCap_snapshot': "Vốn hóa thị trường. Các công ty có vốn hóa lớn thường ổn định và ít rủi ro hơn.",
    'Returns': "Tỷ suất lợi nhuận của cổ phiếu trong một khoảng thời gian."
}

# --- HÀM KẾT NỐI DATABASE ---
@st.cache_resource
def get_db_connection():
    """Tạo kết nối tới MySQL database"""
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='130225',
            database='tickets',
            port=3306
        )
        return connection
    except Error as e:
        st.error(f"❌ Lỗi kết nối database: {e}")
        return None

# --- HÀM LẤY DỮ LIỆU ---
@st.cache_data
def get_stock_data(tickers_list):
    """Lấy các chỉ số tài chính từ Yahoo Finance"""
    financial_data = []
    progress_bar = st.progress(0, text="Đang tải dữ liệu từ Yahoo Finance...")

    for idx, ticker_symbol in enumerate(tickers_list):
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            data = {
                'Mã CP': ticker_symbol, 'Beta': info.get('beta'), 'P/E': info.get('trailingPE'),
                'Nợ/Vốn CSH': info.get('debtToEquity'), 'ROE': info.get('returnOnEquity'),
                'Biên LN': info.get('profitMargins')
            }
            financial_data.append(data)
        except Exception as e:
            st.warning(f"⚠️ Không thể lấy dữ liệu cho '{ticker_symbol}': {e}")
            financial_data.append({
                'Mã CP': ticker_symbol, 'Beta': None, 'P/E': None, 'Nợ/Vốn CSH': None,
                'ROE': None, 'Biên LN': None
            })
        progress_bar.progress((idx + 1) / len(tickers_list))

    progress_bar.empty()
    if not financial_data: return pd.DataFrame()

    df = pd.DataFrame(financial_data).set_index('Mã CP')
    for col in df.columns:
        if df[col].isnull().all():
            st.error(f"❌ Không có dữ liệu cho cột '{col}'")
            return pd.DataFrame()
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
    return df

@st.cache_data
def get_data_from_db(_conn):
    """Lấy dữ liệu từ MySQL database"""
    if _conn and _conn.is_connected():
        try:
            return pd.read_sql("SELECT * FROM tickets_combined", _conn)
        except Exception as e:
            st.error(f"❌ Lỗi truy vấn database: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# === HÀM TÍNH TRỌNG SỐ AHP VÀ CR ===
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
        st.error(f"Lỗi tính toán AHP: {e}")
        return np.array([1/A.shape[0] for _ in range(A.shape[0])]), 1.0


def run_topsis(decision_matrix, weights, impacts):
    """Thuật toán TOPSIS"""
    norm_matrix = decision_matrix / np.sqrt((decision_matrix**2).sum(axis=0))
    weighted_matrix = norm_matrix * weights
    
    ideal_best = np.where(impacts == 1, weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    ideal_worst = np.where(impacts == 1, weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))

    dist_to_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_to_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    
    epsilon = 1e-6
    return dist_to_worst / (dist_to_best + dist_to_worst + epsilon)

# === HÀM TẠO NHẬN XÉT VÀ GỢI Ý ===
def generate_analysis_text(results_df, weights, impacts):
    """Tạo ra các đoạn văn bản phân tích và gợi ý"""
    if results_df.empty: return "", "", ""
    
    best_stock_name = results_df.index[0]
    worst_stock_name = results_df.index[-1]
    best_stock_data = results_df.iloc[0]
    worst_stock_data = results_df.iloc[-1]
    
    # Chuyển đổi impacts từ Benefit/Cost sang dict số (1/-1) nếu cần
    if isinstance(list(impacts.values())[0], str):
        impact_map = {'Benefit': 1, 'Cost': -1}
        numerical_impacts = {k: impact_map[v] for k, v in impacts.items()}
    else:
        numerical_impacts = impacts # Giữ nguyên nếu đã là số

    # Phân tích cổ phiếu tốt nhất
    best_reasons = []
    for crit, impact_val in numerical_impacts.items():
        if crit not in results_df.columns: continue
        val = best_stock_data[crit]
        # Tìm các yếu tố nổi bật (ví dụ: top 25% tốt nhất)
        if (impact_val == 1 and val >= results_df[crit].quantile(0.75)) or \
           (impact_val == -1 and val <= results_df[crit].quantile(0.25)):
            best_reasons.append(f"**{crit}** ({val:.2f})")

    best_analysis = f"🏆 **{best_stock_name}** được xếp hạng cao nhất. Điểm mạnh chính đến từ các chỉ số: {', '.join(best_reasons[:3])}."
    
    # Phân tích cổ phiếu rủi ro nhất
    worst_reasons = []
    for crit, impact_val in numerical_impacts.items():
        if crit not in results_df.columns: continue
        val = worst_stock_data[crit]
        if (impact_val == 1 and val <= results_df[crit].quantile(0.25)) or \
           (impact_val == -1 and val >= results_df[crit].quantile(0.75)):
            worst_reasons.append(f"**{crit}** ({val:.2f})")
    
    worst_analysis = f"⚠️ **{worst_stock_name}** có rủi ro cao nhất trong danh sách, chủ yếu do các chỉ số chưa tốt như: {', '.join(worst_reasons[:3])}."

    # Gợi ý hành động
    actionable_advice = """
    💡 **Gợi ý hành động:**
    - **Đối với các cổ phiếu top đầu:** Đây là những ứng viên sáng giá dựa trên tiêu chí của bạn. Hãy cân nhắc đưa vào danh sách theo dõi và **phân tích sâu hơn** về yếu tố cơ bản của doanh nghiệp trước khi đầu tư.
    - **Đối với các cổ phiếu cuối bảng:** Cần thận trọng với các cổ phiếu này. Nếu đang nắm giữ, bạn nên **xem xét lại vị thế** và có thể đặt các biện pháp phòng ngừa rủi ro như **lệnh cắt lỗ (stop-loss)**.
    - **Lưu ý:** Kết quả này hoàn toàn dựa trên các chỉ số và trọng số bạn đã cung cấp. Đây là công cụ tham khảo, không phải lời khuyên đầu tư trực tiếp.
    """
    
    return best_analysis, worst_analysis, actionable_advice

# === HÀM ĐỊNH DẠNG SỐ LỚN ===
def format_large_number(num):
    """Định dạng số lớn (vd: vốn hóa) thành dạng tỷ, triệu."""
    if pd.isna(num):
        return "N/A"
    num = float(num)
    if num >= 1e12:
        return f"{num / 1e12:.2f} nghìn tỷ"
    if num >= 1e9:
        return f"{num / 1e9:.2f} tỷ"
    if num >= 1e6:
        return f"{num / 1e6:.2f} triệu"
    return f"{num:,.2f}"

# --- GIAO DIỆN CHÍNH ---
st.title("⚖️ Ứng dụng Phân tích Rủi ro Đầu tư Cổ phiếu bằng TOPSIS")
st.info("🎯 Xếp hạng rủi ro cổ phiếu dựa trên các chỉ số tài chính với trọng số tùy chỉnh (có thể dùng AHP để tính trọng số)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Bảng điều khiển")
    data_source = st.radio("📂 Chọn nguồn dữ liệu:", ["Yahoo Finance API", "Database MySQL"], key="data_source")

    weights = {}
    CR = 1.0 # Khởi tạo CR mặc định
    
    if data_source == "Yahoo Finance API":
        criteria_yf = {'Beta': -1, 'P/E': -1, 'Nợ/Vốn CSH': -1, 'ROE': 1, 'Biên LN': 1}
        criteria_list = list(criteria_yf.keys())
        n_criteria = len(criteria_list)
        
        tickers_input = st.text_area("Nhập mã cổ phiếu (phân cách bởi dấu phẩy):", "AAPL, MSFT, GOOGL, TSLA, NVDA")
        
        st.header("⚖️ Thiết lập Trọng số")
        weight_method = st.radio("Chọn phương pháp xác định Trọng số:", ["Sử dụng AHP", "Nhập thủ công"], key="weight_method_yf")

        if weight_method == "Sử dụng AHP":
            st.markdown("Sử dụng thanh trượt để so sánh độ quan trọng của các tiêu chí (1/9...9).")
            
            # 1. Tạo giao diện so sánh cặp AHP
            comparison_values = {}
            for crit_i, crit_j in itertools.combinations(criteria_list, 2):
                key = tuple(sorted((crit_i, crit_j)))
                
                comparison_value = st.select_slider(
                    f"**{crit_i}** quan trọng hơn **{crit_j}** bao nhiêu?",
                    options=[1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    value=1.0,
                    format_func=lambda x: f"1/{round(1/x)}" if x < 1 else str(round(x)),
                    key=f"ahp_{crit_i}_{crit_j}",
                    help=f"So sánh độ quan trọng tương đối: {CRITERIA_EXPLANATIONS.get(crit_i)} so với {CRITERIA_EXPLANATIONS.get(crit_j)}"
                )
                comparison_values[key] = comparison_value
                
            # 2. Xây dựng ma trận so sánh
            comparison_matrix = [[0.0] * n_criteria for _ in range(n_criteria)]
            for i in range(n_criteria):
                for j in range(n_criteria):
                    if i == j:
                        comparison_matrix[i][j] = 1.0
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
            
            if CR > 0.1:
                st.error("⚠️ **CR > 0.1**. Mức độ nhất quán thấp. Vui lòng điều chỉnh các so sánh cặp để đảm bảo kết quả hợp lệ.")
                weights = {}
            else:
                st.success("✅ **CR < 0.1**. Mức độ nhất quán chấp nhận được.")
                weights = {criteria_list[i]: ahp_weights_array[i] for i in range(n_criteria)}
                st.markdown("**Trọng số được tính toán:**")
                weights_df = pd.DataFrame(weights.items(), columns=['Tiêu chí', 'Trọng số']).set_index('Tiêu chí')
                st.dataframe(weights_df.style.format("{:.4f}"))

        else: # Nhập thủ công
            st.subheader("Nhập Trọng số Thủ công")
            default_weights = {k: 0.2 for k in criteria_yf.keys()}
            preset = st.selectbox("🎚️ Chọn bộ trọng số mẫu:", ["Tùy chỉnh", "An toàn (Risk-averse)", "Cân bằng", "Tăng trưởng"], key="preset_yf")

            if preset == "An toàn (Risk-averse)": default_weights = {'Beta': 0.3, 'P/E': 0.2, 'Nợ/Vốn CSH': 0.3, 'ROE': 0.1, 'Biên LN': 0.1}
            elif preset == "Cân bằng": default_weights = {'Beta': 0.2, 'P/E': 0.2, 'Nợ/Vốn CSH': 0.2, 'ROE': 0.2, 'Biên LN': 0.2}
            elif preset == "Tăng trưởng": default_weights = {'Beta': 0.1, 'P/E': 0.1, 'Nợ/Vốn CSH': 0.1, 'ROE': 0.35, 'Biên LN': 0.35}
            else: default_weights = {k: 0.2 for k in criteria_yf.keys()}
            
            for crit, impact in criteria_yf.items():
                weights[crit] = st.slider(
                    f"🎯 {crit}", 0.0, 1.0, default_weights.get(crit, 0.2), 0.05, key=f'w_{crit}_man',
                    help=CRITERIA_EXPLANATIONS.get(crit, "Chưa có giải thích")
                )
            CR = 0.0 # Bỏ qua CR khi nhập thủ công
    
    elif data_source == "Database MySQL":
        criteria_db = DB_CRITERIA
        criteria_list = DB_CRITERIA
        n_criteria = len(criteria_list)
        
        st.header("⚖️ Thiết lập Trọng số")
        st.markdown("Các chỉ số phân tích được lấy mặc định từ database.")
        weight_method = st.radio("Chọn phương pháp xác định Trọng số:", ["Sử dụng AHP", "Nhập thủ công"], key="weight_method_db")
        
        if weight_method == "Sử dụng AHP":
            st.markdown("Sử dụng thanh trượt để so sánh độ quan trọng của các tiêu chí (1/9...9).")
            
            # 1. Tạo giao diện so sánh cặp AHP cho DB
            comparison_values = {}
            for crit_i, crit_j in itertools.combinations(criteria_list, 2):
                key = tuple(sorted((crit_i, crit_j)))
                
                comparison_value = st.select_slider(
                    f"**{crit_i}** (*{DB_IMPACTS[crit_i]}*) quan trọng hơn **{crit_j}** (*{DB_IMPACTS[crit_j]}*) bao nhiêu?",
                    options=[1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    value=1.0,
                    format_func=lambda x: f"1/{round(1/x)}" if x < 1 else str(round(x)),
                    key=f"ahp_db_{crit_i}_{crit_j}",
                    help=f"So sánh độ quan trọng tương đối: {CRITERIA_EXPLANATIONS.get(crit_i)} so với {CRITERIA_EXPLANATIONS.get(crit_j)}"
                )
                comparison_values[key] = comparison_value
                
            # 2. Xây dựng ma trận so sánh
            comparison_matrix = [[0.0] * n_criteria for _ in range(n_criteria)]
            for i in range(n_criteria):
                for j in range(n_criteria):
                    if i == j:
                        comparison_matrix[i][j] = 1.0
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
            
            if CR > 0.1:
                st.error("⚠️ **CR > 0.1**. Mức độ nhất quán thấp. Vui lòng điều chỉnh các so sánh cặp để đảm bảo kết quả hợp lệ.")
                weights = {}
            else:
                st.success("✅ **CR < 0.1**. Mức độ nhất quán chấp nhận được.")
                weights = {criteria_list[i]: ahp_weights_array[i] for i in range(n_criteria)}
                st.markdown("**Trọng số được tính toán:**")
                weights_df = pd.DataFrame(weights.items(), columns=['Tiêu chí', 'Trọng số']).set_index('Tiêu chí')
                st.dataframe(weights_df.style.format("{:.4f}"))
        
        else: # Nhập thủ công
            st.subheader("Nhập Trọng số Thủ công")
            default_weight = 1.0 / len(DB_CRITERIA)
            for crit in DB_CRITERIA:
                weights[crit] = st.slider(
                    f"🎯 {crit} (*{DB_IMPACTS[crit]}*)", 0.0, 1.0, default_weight, 0.05,
                    key=f'w_{crit}_db',
                    help=CRITERIA_EXPLANATIONS.get(crit, "Chưa có giải thích")
                )
            CR = 0.0


    # Kiểm tra tổng trọng số
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ Tổng trọng số: {total_weight:.2f} (Cần = 1.0)")
    else:
        st.success(f"✅ Tổng trọng số: {total_weight:.2f}")

# --- NÚT PHÂN TÍCH ---
if st.button("🚀 Bắt đầu Phân tích", use_container_width=True):
    
    if CR > 0.1:
        st.warning("Vui lòng sửa lỗi Mức độ nhất quán (CR > 0.1) trong bảng điều khiển Trọng số trước khi chạy TOPSIS.")
        st.stop()
        
    # Chuẩn hóa lại trọng số trước khi chạy
    if sum(weights.values()) > 0:
        weights = {k: v / sum(weights.values()) for k, v in weights.items()}
    elif not weights:
         st.error("❌ Không có trọng số hợp lệ để chạy TOPSIS. Vui lòng kiểm tra lại cấu hình.")
         st.stop()
    

    if data_source == "Yahoo Finance API":
        tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        if not tickers_list:
            st.error("❌ Vui lòng nhập ít nhất một mã cổ phiếu")
            st.stop()

        with st.spinner(f"⏳ Đang tải dữ liệu cho {len(tickers_list)} cổ phiếu..."):
            raw_data = get_stock_data(tickers_list)
        
        criteria = {'Beta': -1, 'P/E': -1, 'Nợ/Vốn CSH': -1, 'ROE': 1, 'Biên LN': 1}
        col_impacts = {k: ('Benefit' if v == 1 else 'Cost') for k, v in criteria.items()}

    else: # Database MySQL Flow
        with st.spinner("⏳ Đang kết nối và xử lý dữ liệu từ database..."):
            conn = get_db_connection()
            if not conn: st.stop()
            
            df_raw = get_data_from_db(conn)
            if df_raw.empty: st.stop()

            ticker_col = 'Ticker'
            if ticker_col not in df_raw.columns:
                st.error(f"❌ Không tìm thấy cột '{ticker_col}' trong database.")
                st.stop()

            if 'Date' in df_raw.columns:
                df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
                df_grouped = df_raw.sort_values('Date', ascending=True).groupby(ticker_col).last().reset_index()
            else:
                df_grouped = df_raw.drop_duplicates(subset=[ticker_col], keep='last').reset_index(drop=True)
            
            missing_cols = [col for col in DB_CRITERIA if col not in df_grouped.columns]
            if missing_cols:
                st.error(f"❌ Database thiếu các cột bắt buộc: **{', '.join(missing_cols)}**")
                st.stop()

            st.success(f"✅ Đã xử lý **{len(df_grouped)}** mã cổ phiếu với {len(DB_CRITERIA)} chỉ số mặc định.")
        
        raw_data = df_grouped.set_index(ticker_col)[DB_CRITERIA].copy()
        raw_data.index.name = 'Mã CP'

        for col in raw_data.columns:
            raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')
            if raw_data[col].isnull().any():
                median_val = raw_data[col].median()
                raw_data[col] = raw_data[col].fillna(median_val if pd.notna(median_val) else 0)

        criteria = {col: (1 if DB_IMPACTS[col] == 'Benefit' else -1) for col in DB_CRITERIA}
        col_impacts = DB_IMPACTS

    if raw_data.empty:
        st.error("❌ Không có dữ liệu để phân tích.")
        st.stop()

    # === PHẦN SỬA ĐỔI: Áp dụng định dạng số lớn cho bảng dữ liệu đầu vào ===
    st.header("📊 Dữ liệu Đầu vào sau khi xử lý")
    formatters = {col: "{:,.2f}" for col in raw_data.columns}
    if 'marketCap_snapshot' in raw_data.columns:
        formatters['marketCap_snapshot'] = format_large_number
    st.dataframe(raw_data.style.format(formatters).background_gradient(cmap='YlOrRd', axis=0), use_container_width=True)

    st.header("🏆 Kết quả Xếp hạng TOPSIS")
    decision_matrix = raw_data[list(criteria.keys())]
    weights_list = np.array([weights[crit] for crit in criteria])
    impacts_list = np.array([criteria[crit] for crit in criteria])

    scores = run_topsis(decision_matrix, weights_list, impacts_list)

    results_df = raw_data.copy()
    results_df['Điểm TOPSIS'] = scores
    results_df['Xếp hạng'] = results_df['Điểm TOPSIS'].rank(ascending=False).astype(int)
    results_df = results_df.sort_values(by='Xếp hạng')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🥇 Tốt nhất (Ít rủi ro nhất)", results_df.index[0], f"{results_df['Điểm TOPSIS'].iloc[0]:.4f}")
    col2.metric("🥈 Rủi ro nhất", results_df.index[-1], f"{results_df['Điểm TOPSIS'].iloc[-1]:.4f}")
    col3.metric("📊 Điểm trung bình", f"{results_df['Điểm TOPSIS'].mean():.4f}")
    
    # === PHẦN SỬA ĐỔI: Áp dụng định dạng số lớn cho bảng kết quả ===
    result_formatters = {col: "{:,.4f}" for col in raw_data.columns}
    result_formatters['Điểm TOPSIS'] = "{:.4f}"
    if 'marketCap_snapshot' in results_df.columns:
        result_formatters['marketCap_snapshot'] = format_large_number

    st.dataframe(
        results_df.style
        .apply(lambda row: ['background-color: #2E8B57; color: white; font-weight: bold' if row['Xếp hạng'] == 1 else '' for _ in row], axis=1)
        .background_gradient(cmap='Greens', subset=['Điểm TOPSIS'])
        .format(result_formatters),
        use_container_width=True
    )
    
    # === PHẦN MỚI: Hiển thị phân tích và gợi ý ===
    st.markdown("---")
    st.subheader("📝 Phân tích và Gợi ý từ Hệ thống")
    best_analysis, worst_analysis, actionable_advice = generate_analysis_text(results_df, weights, col_impacts)
    st.markdown(best_analysis)
    st.markdown(worst_analysis)
    st.markdown(actionable_advice)
    st.markdown("---")


    tab1, tab2, tab3 = st.tabs(["📊 Biểu đồ So sánh", "🎯 Biểu đồ Radar", "📥 Tải xuống"])
    with tab1:
        fig = px.bar(results_df, x=results_df.index, y='Điểm TOPSIS', color='Điểm TOPSIS',
                     color_continuous_scale='Greens', title='So sánh Điểm TOPSIS giữa các Cổ phiếu', text='Điểm TOPSIS')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(xaxis_title="Mã Cổ phiếu", yaxis_title="Điểm TOPSIS", xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        top_n = min(5, len(results_df))
        top_stocks = results_df.head(top_n)
        norm_data = (top_stocks[list(criteria.keys())] - top_stocks[list(criteria.keys())].min()) / \
                    (top_stocks[list(criteria.keys())].max() - top_stocks[list(criteria.keys())].min())
        fig = go.Figure()
        for ticker in top_stocks.index:
            fig.add_trace(go.Scatterpolar(r=norm_data.loc[ticker].values, theta=list(criteria.keys()), fill='toself', name=ticker))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title=f"So sánh Top {top_n} Cổ phiếu (dữ liệu đã chuẩn hóa)")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Ket_qua_TOPSIS')
            raw_data.to_excel(writer, sheet_name='Du_lieu_goc')
            pd.DataFrame({
                'Chỉ số': list(weights.keys()),
                'Trọng số': list(weights.values()),
                'Tác động': [col_impacts[k] for k in weights.keys()]
            }).to_excel(writer, sheet_name='Cau_hinh_phan_tich', index=False)
        
        st.download_button("📊 Tải báo cáo chi tiết (.xlsx)", output.getvalue(),
                          f"topsis_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

st.markdown("---")
st.markdown("🔬 **TOPSIS Analysis System**")
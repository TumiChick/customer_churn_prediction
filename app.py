import streamlit as st
import pandas as pd
import joblib


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# 2. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")


model = load_model()


# =========================================================
# 3. HEADER
# =========================================================

st.title("📊 Customer Churn Prediction")

st.write(
    """
    Nhập thông tin khách hàng để dự đoán khả năng khách hàng
    rời bỏ dịch vụ (**Customer Churn**).
    """
)

st.info(
    "Mô hình sử dụng Logistic Regression để ước lượng "
    "xác suất Churn của khách hàng."
)


# =========================================================
# 4. INPUT FORM
# =========================================================

with st.form("churn_form"):

    # -----------------------------------------------------
    # CUSTOMER INFORMATION
    # -----------------------------------------------------

    st.subheader("👤 Thông tin khách hàng")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

    with col2:

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

        tenure_months = st.number_input(
            "Tenure Months",
            min_value=0,
            max_value=72,
            value=12,
            step=1
        )

    st.divider()

    # -----------------------------------------------------
    # SERVICES
    # -----------------------------------------------------

    st.subheader("🌐 Dịch vụ")

    col1, col2 = st.columns(2)

    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    st.divider()

    # -----------------------------------------------------
    # CONTRACT AND PAYMENT
    # -----------------------------------------------------

    st.subheader("💳 Hợp đồng và thanh toán")

    col1, col2 = st.columns(2)

    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col2:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=18.25,
            max_value=118.75,
            value=70.00,
            step=0.05
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=8684.80,
            value=500.00,
            step=10.0
        )

        cltv = st.number_input(
            "CLTV",
            min_value=2003.0,
            max_value=6500.0,
            value=4000.0,
            step=10.0
        )

    st.divider()

    # -----------------------------------------------------
    # SUBMIT
    # -----------------------------------------------------

    submitted = st.form_submit_button(
        "🔍 Predict Churn",
        use_container_width=True
    )


# =========================================================
# 5. VALIDATION
# =========================================================

if submitted:

    errors = []

    # -----------------------------------------------------
    # Phone Service validation
    # -----------------------------------------------------

    if (
        phone_service == "No"
        and multiple_lines != "No phone service"
    ):
        errors.append(
            "Nếu Phone Service = No thì Multiple Lines "
            "phải là 'No phone service'."
        )

    if (
        phone_service == "Yes"
        and multiple_lines == "No phone service"
    ):
        errors.append(
            "Nếu Phone Service = Yes thì Multiple Lines "
            "không thể là 'No phone service'."
        )

    # -----------------------------------------------------
    # Internet Service validation
    # -----------------------------------------------------

    internet_features = [
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies
    ]

    if internet_service == "No":

        if any(
            value != "No internet service"
            for value in internet_features
        ):
            errors.append(
                "Nếu Internet Service = No thì Online Security, "
                "Online Backup, Device Protection, Tech Support, "
                "Streaming TV và Streaming Movies phải là "
                "'No internet service'."
            )

    else:

        if any(
            value == "No internet service"
            for value in internet_features
        ):
            errors.append(
                "Nếu khách hàng có Internet Service thì các dịch vụ "
                "Internet không thể có giá trị "
                "'No internet service'."
            )

    # -----------------------------------------------------
    # Display validation errors
    # -----------------------------------------------------

    if errors:

        st.subheader("⚠️ Dữ liệu chưa hợp lệ")

        for error in errors:
            st.error(error)

        st.stop()


    # =====================================================
    # 6. CREATE CUSTOMER DATAFRAME
    # =====================================================

    customer = pd.DataFrame([{

        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,

        "Tenure Months": tenure_months,

        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,

        "Internet Service": internet_service,

        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,

        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,

        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,

        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "CLTV": cltv

    }])


    # =====================================================
    # 7. MODEL PREDICTION
    # =====================================================

    try:

        prediction = model.predict(
            customer
        )[0]

        churn_probability = model.predict_proba(
            customer
        )[0, 1]

    except Exception as e:

        st.error(
            "Có lỗi xảy ra khi thực hiện dự đoán."
        )

        st.exception(e)

        st.stop()


    # =====================================================
    # 8. RESULT
    # =====================================================

    st.divider()

    st.header("📈 Kết quả dự đoán")

    result_col1, result_col2 = st.columns(2)


    # -----------------------------------------------------
    # Prediction label
    # -----------------------------------------------------

    with result_col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2%}"
        )


    # -----------------------------------------------------
    # Risk level
    # -----------------------------------------------------

    if churn_probability >= 0.70:

        risk_level = "High Risk"

    elif churn_probability >= 0.40:

        risk_level = "Medium Risk"

    else:

        risk_level = "Low Risk"


    with result_col2:

        st.metric(
            "Risk Level",
            risk_level
        )


    # -----------------------------------------------------
    # Prediction message
    # -----------------------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ Mô hình dự đoán khách hàng có nguy cơ CHURN."
        )

    else:

        st.success(
            "✅ Mô hình dự đoán khách hàng có khả năng NO CHURN."
        )


    # -----------------------------------------------------
    # Probability progress bar
    # -----------------------------------------------------

    st.write("#### Mức độ rủi ro Churn")

    st.progress(
        float(churn_probability)
    )

    st.caption(
        f"Estimated Churn Probability: "
        f"{churn_probability:.2%}"
    )


    # =====================================================
    # 9. BUSINESS INTERPRETATION
    # =====================================================

    st.subheader("💡 Gợi ý")

    if risk_level == "High Risk":

        st.warning(
            """
            Khách hàng có mức rủi ro churn cao.

            Doanh nghiệp có thể ưu tiên khách hàng này
            trong các hoạt động retention hoặc chăm sóc
            khách hàng chủ động.
            """
        )

    elif risk_level == "Medium Risk":

        st.info(
            """
            Khách hàng có mức rủi ro churn trung bình.

            Có thể tiếp tục theo dõi và cân nhắc các
            chương trình chăm sóc hoặc ưu đãi phù hợp.
            """
        )

    else:

        st.success(
            """
            Khách hàng hiện có mức rủi ro churn thấp.

            Có thể tiếp tục theo dõi bằng quy trình
            chăm sóc khách hàng thông thường.
            """
        )


    # =====================================================
    # 10. INPUT DATA PREVIEW
    # =====================================================

    with st.expander(
        "🔎 Xem dữ liệu đầu vào"
    ):

        st.dataframe(
            customer,
            use_container_width=True
        )


# =========================================================
# 11. DISCLAIMER
# =========================================================

st.divider()

st.caption(
    """
    Lưu ý: Các mức Low Risk, Medium Risk và High Risk hiện được
    thiết lập phục vụ mục đích minh họa. Các ngưỡng 40% và 70%
    chưa được tối ưu dựa trên chi phí nghiệp vụ hoặc threshold
    optimization. Kết quả của mô hình nên được sử dụng như một
    công cụ hỗ trợ ra quyết định.
    """
)
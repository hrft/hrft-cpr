# 📈 راهنمای اجرای پروژه HRFT-CPR (Crypto Prediction Dashboard)

این راهنما به شما کمک می‌کند پروژه را روی کامپیوتر خود اجرا کنید.

---

## 1. آماده‌سازی

### 1.1 نصب پایتون
- مطمئن شوید Python نسخه 3.9 یا بالاتر روی سیستم شما نصب است.
- تست نصب:
```bash
python --version

1.2 ورود به پوشه پروژه

در ترمینال (CMD, PowerShell یا Terminal):

cd path/to/hrft-cpr


(به جای path/to/ مسیر واقعی پوشه را وارد کنید.)

2. ساخت محیط مجازی (اختیاری ولی توصیه‌شده)
ویندوز:
python -m venv venv
venv\Scripts\activate

مک/لینوکس:
python -m venv venv
source venv/bin/activate

3. نصب پیش‌نیازها
pip install -r requirements.txt

4. اجرای پروژه
streamlit run app.py


پس از اجرا، لینکی شبیه زیر در ترمینال نمایش داده می‌شود:

Local URL: http://localhost:8501

5. استفاده از داشبورد

از منوی سمت چپ:

انتخاب ارز دیجیتال (BTC, ETH, BNB, SOL)

انتخاب بازه زمانی (1d = روزانه، 1h = ساعتی، 30m = نیم‌ساعتی)

قیمت لحظه‌ای از CoinGecko نمایش داده می‌شود.

نمودار اصلی:

نمودار کندلی (قیمت واقعی)

خط قرمز (پیش‌بینی ۷ روز آینده)

6. بروزرسانی و تغییرات

هر بار که فایل app.py را تغییر دهید:

در ترمینال Ctrl + C بزنید تا برنامه متوقف شود

دوباره اجرا کنید:

streamlit run app.py


یا فقط مرورگر را Refresh کنید.


---

✅ حالا فقط کافیه این فایل‌ها رو بذاری توی پوشه `hrft-cpr`.  
بعد بزنی:
```bash
streamlit run app.py

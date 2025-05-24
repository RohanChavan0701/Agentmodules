from flask import Flask, request, jsonify
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import base64
import os
import time

app = Flask(__name__)
CORS(app)

# Initialize Chrome driver with options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = None

def get_driver():
    global driver
    if driver is None:
        driver = webdriver.Chrome(options=chrome_options)
    return driver

@app.route('/navigate', methods=['POST'])
def navigate():
    try:
        data = request.json
        url = data.get('url')
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        browser = get_driver()
        browser.get(url)
        time.sleep(2)  # Wait for page to load
        return jsonify({"message": "Navigation successful"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/screenshot', methods=['GET'])
def screenshot():
    try:
        browser = get_driver()
        # Take screenshot
        screenshot = browser.get_screenshot_as_base64()
        return jsonify({
            "message": "Screenshot captured",
            "data": screenshot
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
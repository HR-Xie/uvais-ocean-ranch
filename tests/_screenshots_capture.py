"""Capture full-screen screenshots for README"""
import sys, time, re
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8501"
DIR = "docs/screenshots"

import os
os.makedirs(DIR, exist_ok=True)

def log(msg):
    safe = msg.encode('ascii', errors='replace').decode('ascii')
    print(safe, flush=True)

def send(page, question, wait_s=60):
    """Send question and wait for response"""
    input_el = page.locator('[data-testid="stChatInputTextArea"]')
    input_el.click()
    input_el.fill(question)
    page.keyboard.press("Enter")
    time.sleep(wait_s)

def snap(page, name):
    path = f"{DIR}/{name}.png"
    page.screenshot(path=path, full_page=True)
    log(f"  Screenshot: {name}.png")

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=False, slow_mo=150)
    page = browser.new_page(viewport={"width": 1440, "height": 900})

    # ==================== 1. Welcome / Main UI (Manager) ====================
    log("\n=== 1. Main UI Welcome ===")
    page.goto(BASE, timeout=30000)
    page.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=15000)
    time.sleep(2)
    snap(page, "01_main_ui")

    # ==================== 2. RAG Knowledge Retrieval ====================
    log("\n=== 2. RAG Knowledge ===")
    send(page, "浪高超过多少米禁止潜水作业", wait_s=50)
    snap(page, "02_rag_knowledge")

    # ==================== 3. Sea State Query (Agent Tool) ====================
    log("\n=== 3. Sea State ===")
    # Clear by reloading
    page.goto(BASE, timeout=30000)
    page.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=15000)
    time.sleep(1)
    send(page, "查询C区网箱的实时海况", wait_s=35)
    snap(page, "03_sea_state")

    # ==================== 4. Vision Model Query ====================
    log("\n=== 4. Vision Model ===")
    page.goto(BASE, timeout=30000)
    page.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=15000)
    time.sleep(1)
    send(page, "检查水下摄像头img_001的画面有没有异常", wait_s=35)
    snap(page, "04_vision_model")

    # ==================== 5. Worker Navigation ====================
    log("\n=== 5. Worker Navigation ===")
    # Switch to worker role
    page.goto(f"{BASE}/?preview=worker", timeout=30000)
    page.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=15000)
    time.sleep(2)
    # Click on worker radio if available
    send(page, "获取当前位置与测距", wait_s=35)
    snap(page, "05_worker_nav")

    # ==================== 6. Dispatch + Work Order ====================
    log("\n=== 6. Dispatch Work Order ===")
    page.goto(f"{BASE}/?preview=manager", timeout=30000)
    page.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=15000)
    time.sleep(1)
    # First get sea state
    send(page, "查询安全海域的海况数据", wait_s=40)
    # Then dispatch
    send(page, "海况确认安全了，生成一份C区网箱巡检工单报告", wait_s=50)
    snap(page, "06_work_order")

    # ==================== 7. Multi-turn Conversation ====================
    log("\n=== 7. Multi-turn ===")
    page.goto(BASE, timeout=30000)
    page.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=15000)
    time.sleep(1)
    send(page, "禁止人工潜水作业的浪高条件是什么", wait_s=40)
    send(page, "那流速的限制是多少", wait_s=35)
    snap(page, "07_multiturn")

    # ==================== 8. Sidebar detail ====================
    log("\n=== 8. Sidebar ===")
    snap(page, "08_sidebar_detail")

    browser.close()

log("\nAll screenshots captured!")
log(f"Saved to {DIR}/")

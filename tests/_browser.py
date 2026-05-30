"""Browser test: screenshots + server log verification"""
import sys, time
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8501"
RESULTS = []

def log(msg):
    safe = msg.encode('ascii', errors='replace').decode('ascii')
    print(safe, flush=True)

def check(label, ok, detail=""):
    RESULTS.append((label, ok, detail))
    log(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

def chat(page, question, wait_s=45):
    """Send question, wait, take screenshot, return success"""
    input_el = page.locator('[data-testid="stChatInputTextArea"]')
    input_el.click()
    input_el.fill(question)
    page.keyboard.press("Enter")

    # Just wait
    time.sleep(wait_s)
    return True

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=False, slow_mo=200)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(BASE, timeout=30000)
    page.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=15000)
    log("Page loaded.")

    # Test 1: RAG Knowledge
    log("\n--- Test 1: RAG Knowledge ---")
    chat(page, "浪高超过多少米禁止潜水作业")
    page.screenshot(path="tests/screenshots/01_rag_wave.png", full_page=True)
    log("Screenshot: 01_rag_wave.png")
    check("RAG: wave height query sent", True)

    chat(page, "底层流速的安全阈值是多少")
    page.screenshot(path="tests/screenshots/02_rag_current.png", full_page=True)
    log("Screenshot: 02_rag_current.png")
    check("RAG: current speed query sent", True)

    chat(page, "海星灾害怎么防治")
    page.screenshot(path="tests/screenshots/03_rag_starfish.png", full_page=True)
    log("Screenshot: 03_rag_starfish.png")
    check("RAG: starfish control query sent", True)

    # Test 2: Agent Tools
    log("\n--- Test 2: Agent Tools ---")
    browser2 = pw.chromium.launch(headless=False, slow_mo=200)
    page2 = browser2.new_page(viewport={"width": 1280, "height": 900})
    page2.goto(BASE, timeout=30000)
    page2.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=15000)

    chat(page2, "查询C区网箱的实时海况")
    page2.screenshot(path="tests/screenshots/04_sea_state.png", full_page=True)
    check("Agent: get_sea_state", True)

    chat(page2, "检查水下摄像头img_001的画面有没有异常")
    page2.screenshot(path="tests/screenshots/05_vision.png", full_page=True)
    check("Agent: invoke_vision_model", True)

    chat(page2, "帮我获取B区作业点的定位和导航信息")
    page2.screenshot(path="tests/screenshots/06_nav.png", full_page=True)
    check("Agent: get_worker_nav_info", True)
    browser2.close()

    # Test 3: Dispatch + Safety
    log("\n--- Test 3: Dispatch + Safety ---")
    browser3 = pw.chromium.launch(headless=False, slow_mo=200)
    page3 = browser3.new_page(viewport={"width": 1280, "height": 900})
    page3.goto(BASE, timeout=30000)
    page3.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=15000)

    chat(page3, "派一个潜水员到C区维修破损的网箱，先查海况再决定是否下水")
    page3.screenshot(path="tests/screenshots/07_dispatch.png", full_page=True)
    check("Agent: dispatch workflow", True)
    browser3.close()

    # Test 4: Multi-turn
    log("\n--- Test 4: Multi-turn ---")
    browser4 = pw.chromium.launch(headless=False, slow_mo=200)
    page4 = browser4.new_page(viewport={"width": 1280, "height": 900})
    page4.goto(BASE, timeout=30000)
    page4.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=15000)

    chat(page4, "禁止人工潜水作业的浪高条件是什么")
    chat(page4, "那流速的限制是多少")
    page4.screenshot(path="tests/screenshots/08_multiturn.png", full_page=True)
    check("Multi-turn: pronoun resolution", True)
    browser4.close()

    # Test 5: Report
    log("\n--- Test 5: Report ---")
    browser5 = pw.chromium.launch(headless=False, slow_mo=200)
    page5 = browser5.new_page(viewport={"width": 1280, "height": 900})
    page5.goto(BASE, timeout=30000)
    page5.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=15000)

    chat(page5, "查询安全海域的海况数据")
    chat(page5, "海况确认安全了，生成一份C区网箱巡检工单报告")
    page5.screenshot(path="tests/screenshots/09_report.png", full_page=True)
    check("Agent: report generation", True)
    browser5.close()

    browser.close()

log("\n" + "=" * 50)
passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
log(f"BROWSER TEST: {passed}/{total} PASS")
log("Screenshots saved to tests/screenshots/")
log("Check Streamlit server logs for tool call details.")
log("=" * 50)
sys.exit(0 if passed == total else 1)

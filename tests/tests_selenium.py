from selenium import webdriver

# Specify the path to the WebDriver
driver = webdriver.Chrome(executable_path='path_to_your_chromedriver')

# Open a website
driver.get('https://www.example.com')

# Perform actions (e.g., finding an element, clicking, etc.)
# element = driver.find_element_by_name('q')  # Example for searching
# element.send_keys('Selenium')
# element.submit()

# Close the browser
driver.quit()

from datetime import datetime, timedelta


now = datetime.now().date()
start = datetime.now().date() - timedelta(12)

print('============================')
print(start, now)
print('Y-FoRM 안녕?')
print('============================')

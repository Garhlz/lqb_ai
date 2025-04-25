02 电影信息提取

# 一、任务介绍
在互联网时代，海量的电影和电视节目信息以网页形式呈现，网页中涵盖了电影的标题、类别、摘要、上映年份等关键信息。但这些信息往往嵌入在复杂的 HTML 结构里，这使得自动提取和分析这些数据颇具挑战。掌握从网页中提取结构化数据的技术，对数据挖掘和信息获取意义重大。本任务旨在运用网页分析技术，解析包含电影信息的 HTML 页面，提取并还原电影的标题、类别、摘要和上映年份，最终生成一个结构化的数据文件。

# 二、准备工作
在开始答题前，请确认 /home/project/02 目录下包含以下文件：
html_pages/*.html：这是本任务提供的网页文件，共有 3 种类型：
网站首页，文件名为 html_pages/index.html。
类别页面，其文件地址由首页提供。
详情页面，其文件地址由类别页面提供。
task.py：这是你后续答题时编写代码的文件。源代码的运行方式如下：

# 三、任务目标
请按照要求实现以下目标：
## 目标 1：实现 parse_index_page 函数
参数：
index_path：首页 HTML 文件的路径。
功能：解析首页，提取所有类别页面的链接。
返回值：一个列表，包含每个类别页面的链接。
## 目标 2：实现 parse_category_page 函数
参数：
category_page_path：类别页面的文件路径。
html_dir：HTML 文件的存放目录，用于构造完整的电影详情页面路径。
功能：解析类别页面，提取该类别中所有电影的链接。
返回值：一个包含电影详情页面链接的列表。
## 目标 3：实现 parse_movie_page 函数
参数：
movie_page_path：电影详情页面的文件路径。
功能：解析电影详情页面，提取并返回电影的标题、摘要、类别和年份。
返回值：包含电影标题、摘要、类别和年份的元组。
## 目标 4：实现 save_to_csv 函数
参数：
movie_data：要保存的电影数据列表，每个元组包含电影的标题、摘要、类别和年份。
output_csv：输出的 CSV 文件路径。
功能：将提取的电影信息写入指定的 CSV 文件。
请基于以下代码补充 #TODO 处的函数代码，并执行 main() 函数，以确保能够实现上述目标。

# 四、实现提示
创建对象：使用 BeautifulSoup(file, 'html.parser') 解析 HTML 文件。
查找元素：
使用 find('div', class_='categories') 定位容器。
使用 find_all('a') 提取链接。
使用 find('h1', class_='title') 提取电影信息。
提取数据：
使用 get('href') 获取链接地址。
使用 get_text(strip=True) 获取文本内容。
健壮性：
检查 None（如 if title else ""）。
使用 encoding='utf-8' 处理编码。
写入文件：
```python
with open() as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # 写入表头
    writer.writerows(movie_data)  # 写入数据
```
# 电影信息提取, 使用 BeautifulSoup 解析 HTML 文件
import csv
import os
from typing import List, Tuple

from bs4 import BeautifulSoup


def parse_index_page(
    index_path: str,
) -> List[str]:
    """
    解析首页 HTML 文件，提取所有类别页面的链接。

    Args:
        index_path (str): 首页 HTML 文件路径。

    Returns:
        List[str]: 类别页面链接列表。
    """
    # 读取首页 HTML 文件
    with open(index_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # 查找包含类别链接的元素（假设在 <a> 标签的 href 属性中）
    category_links = []
    # 假设类别链接在 class="categories" 的 div 中
    categories_div = soup.find("div", class_="categories")
    if categories_div:
        for link in categories_div.find_all("a"):
            href = link.get("href")
            if href:  # 确保 href 不为空
                category_links.append(href)

    return category_links


def parse_category_page(category_page_path: str, html_dir: str) -> List[str]:
    """
    解析类别页面 HTML 文件，提取所有电影详情页面的链接。

    Args:
        category_page_path (str): 类别页面文件路径。
        html_dir (str): HTML 文件存放目录，用于构造完整路径。

    Returns:
        List[str]: 电影详情页面链接列表。
    """
    # 读取类别页面 HTML 文件
    with open(category_page_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # 查找包含电影链接的元素
    movie_links = []
    # 假设电影链接在 class="movies" 的 div 中
    movies_div = soup.find("div", class_="movies")
    if movies_div:
        for link in movies_div.find_all("a"):
            href = link.get("href")
            if href:  # 确保 href 不为空
                # 返回相对路径（相对于 html_dir）
                movie_links.append(href)

    return movie_links


def parse_movie_page(
    movie_page_path: str,
) -> Tuple[str, str, str, str]:
    """
    解析电影详情页面，提取标题、摘要、类别和年份。

    Args:
        movie_page_path (str): 电影详情页面文件路径。

    Returns:
        Tuple[str, str, str, str]: 包含标题、摘要、类别和年份的元组。
    """
    # 读取电影详情页面 HTML 文件
    with open(movie_page_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # 提取标题（假设在 <h1 class="title"> 中）
    title = soup.find("h1", class_="title")
    title = title.get_text().strip() if title else ""

    # 提取摘要（假设在 <p class="summary"> 中）
    summary = soup.find("p", class_="summary")
    summary = summary.get_text().strip() if summary else ""

    # 提取类别（假设在 <span class="genre"> 中）
    genre = soup.find("span", class_="genre")
    genre = genre.get_text().strip() if genre else ""

    # 提取年份（假设在 <span class="year"> 中）
    year = soup.find("span", class_="year")
    year = year.get_text().strip() if year else ""

    return (title, summary, genre, year)


def save_to_csv(
    movie_data: List[Tuple[str, str, str, str]],
    output_csv: str,
) -> None:
    """
    将电影信息保存到 CSV 文件。

    Args:
        movie_data (List[Tuple[str, str, str, str]]): 电影信息列表，每个元组包含标题、摘要、类别和年份。
        output_csv (str): 输出 CSV 文件路径。
    """
    # 定义 CSV 表头
    headers = [
        "Title",
        "Summary",
        "Genre",
        "Year",
    ]

    # 写入 CSV 文件
    with open(
        output_csv,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(headers)
        # 写入电影数据
        writer.writerows(movie_data)


def main(html_dir: str, output_csv: str) -> None:
    index_path = os.path.join(html_dir, "index.html")
    category_links = parse_index_page(index_path)

    movie_data = []
    for category_link in category_links:
        category_page_path = os.path.join(html_dir, category_link)
        movie_links = parse_category_page(category_page_path, html_dir)
        for movie_link in movie_links:
            movie_page_path = os.path.join(html_dir, movie_link)
            movie_info = parse_movie_page(movie_page_path)
            movie_data.append(movie_info)

    save_to_csv(movie_data, output_csv)


if __name__ == "__main__":
    html_dir = "/home/project/02/html_pages"
    output_csv = "/home/project/02/movie_data.csv"
    main(html_dir, output_csv)

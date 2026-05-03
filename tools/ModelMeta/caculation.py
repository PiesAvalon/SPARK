import re
from bs4 import BeautifulSoup
import os
import csv
#csv_file_path = "/home/cvgroup/myz/wr/eagle-main/EAGLE/mnt/equivalentmodels_data/test_cases1.csv"
import argparse

def calculate_indentation(file_path):
    """
    计算指定 .py 文件中每一行的缩进空格数，并根据缩进判断函数体范围。

    :param file_path: 文件路径
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        all_def = {}
        def_stack = []  # 栈，用于追踪嵌套的def函数

        for line_number, line in enumerate(lines, start=1):
            # 去掉行尾的空格或换行符
            stripped_line = line.rstrip()

            # 如果行不为空（避免空行干扰缩进判断）
            if stripped_line:
                # 计算前导空格数
                leading_spaces = len(line) - len(line.lstrip(' '))

                # 判断是否是函数定义
                if stripped_line.lstrip().startswith('def') and len(def_stack) == 0:
                    # 提取函数名
                    func_name = stripped_line.lstrip()[4:].split('(')[0].strip()
                    def_stack.append({'indent': leading_spaces, 'start_line': line_number, 'name': func_name})

                elif stripped_line.lstrip().startswith('def'):
                    if def_stack:
                        current_def = def_stack[-1]
                        if leading_spaces == current_def['indent']:
                            all_def[current_def['name']] = [current_def['start_line'], line_number - 1]
                            def_stack.pop()
                            func_name = stripped_line.lstrip()[4:].split('(')[0].strip()
                            def_stack.append({'indent': leading_spaces, 'start_line': line_number, 'name': func_name})

                # 判断是否是return语句
                elif stripped_line.lstrip().startswith('return'):
                    if def_stack:
                        current_def = def_stack[-1]
                        # 判断是否为当前函数的return
                        if leading_spaces <= current_def['indent']:
                            all_def[current_def['name']] = [current_def['start_line'], line_number]
                            def_stack.pop()
                        elif leading_spaces == current_def['indent'] + 4:
                            all_def[current_def['name']] = [current_def['start_line'], line_number]
                            def_stack.pop()

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查路径！")
    except Exception as e:
        print(f"发生错误：{e}")
    return all_def


def parse_coverage_report(html_content):
    """
    从 coverage index.html 中提取出 {子页面路径: py文件路径} 的映射关系
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    #print('soup',soup)
    report_data = {}
    file_rows = soup.find_all('tr', class_='region')

    
    for row in file_rows:
        file_link_td = row.find('td', class_='name left')
        if not file_link_td:
            continue

        file_link = file_link_td.find('a')
        if file_link:
            file_key = file_link.get('href')  # 如 'example_py.html'
            # 将链接文字视为 .py 文件路径
            py_file_path = file_link.get_text()  # 如 'example.py'
            report_data[file_key] = py_file_path
    
    return report_data




# parser = argparse.ArgumentParser()
# parser.add_argument("script_path", type=str)
# # parser.add_argument("gen_time",  type=str)
# # parser.add_argument("execution_time", type=str)
# args = parser.parse_args()



#---------------------------------------------------------
#   1) 这里是多个测试用例的 index.html 和对应的 htmlcov 目录
#   你可以添加更多条目到这个列表
#---------------------------------------------------------
scenarios = [
    {
        'index_html': '/home/cvgroup/myz/czx/semtest-gitee/modelmeta/htmlcov/index.html',
        'htmlcov_dir': '/home/cvgroup/myz/czx/semtest-gitee/modelmeta/htmlcov/'
    }
    # 
    # {
    #     'index_html': '/home/cvgroup/myz/czx/json2msmodel/htmlcov/index.html',
    #     'htmlcov_dir': '/home/cvgroup/myz/czx/json2msmodel/htmlcov/'
    # },
    # 也可以继续加更多测试用例...
]

#---------------------------------------------------------
#   用于累加所有测试用例的最终结果
#   final_result = {
#       'some.py': [总函数数之和, 被覆盖的函数数之和],
#       ...
#   }
#---------------------------------------------------------
final_result = {}


# 用于存储所有测试用例的函数名称
all_total_function_names = []
all_covered_function_names = []
#---------------------------------------------------------
#   2) 开始循环每个测试用例，逐个解析并统计覆盖
#---------------------------------------------------------
for scenario in scenarios:
    # 读取当前测试用例的 index.html
    with open(scenario['index_html'], 'r', encoding='utf-8') as f:
        html_content = f.read()
    #print('html_content',html_content)
    # 解析当前测试用例的报表数据
    coverage_data = parse_coverage_report(html_content)
    #print('coverage_data:',coverage_data)
    # coverage_result 只用于保存本测试用例下的结果
    coverage_result = {}

    # 遍历 index.html 里列出的每个文件
    for html_file_name, py_file_name in coverage_data.items():
        def_file_path = py_file_name  # 真实的 .py 文件
        all_def = calculate_indentation(def_file_path)

        # 将本文件中的所有函数名称加入总列表
        all_total_function_names.extend(list(all_def.keys()))
        
        total_functions = len(all_def)
        covered_functions = 0

        # 得到对应的 HTML 报告文件的完整路径（如 /xxx/htmlcov/abc_py.html）
        full_html_path = scenario['htmlcov_dir'] + html_file_name

        # 解析覆盖率详情
        try:
            with open(full_html_path, 'r', encoding='utf-8') as file:
                detail_html = file.read()

            soup_detail = BeautifulSoup(detail_html, 'lxml')
            main_content = soup_detail.find('main', id='source')
            if not main_content:
                # 若 html 结构意外变化，跳过
                coverage_result[def_file_path] = [total_functions, 0]
                continue

            lines = main_content.find_all('p')

            # 遍历每个函数，判断是否覆盖
            for func_name, (start_line, end_line) in all_def.items():
                for code_line_index in range(start_line + 1, end_line):
                    # 检查该行是否标记为 class="run" 并且不是装饰器行
                    if ('class="run"' in str(lines[code_line_index]) 
                        and not lines[code_line_index].text.strip().startswith('@')):
                        covered_functions += 1
                        # 将覆盖的函数名称加入列表
                        func_name = def_file_path.replace("/home/cvgroup/miniconda3/envs/czx/lib/python3.9/site-packages/",'').replace("/",'.').replace(".py",'.') + func_name
                        all_covered_function_names.append(func_name)
                        break

        except FileNotFoundError:
            print(f"HTML 文件 {full_html_path} 未找到，请检查路径！")
        except Exception as e:
            print(f"分析 {full_html_path} 时发生错误：{e}")

        coverage_result[def_file_path] = [total_functions, covered_functions]
    #print(coverage_result)
    #---------------------------------------------------------
    #   3) 将当前 coverage_result 累加到最终的 final_result
    #---------------------------------------------------------
    for k, v in coverage_result.items():


        if k not in final_result:
            final_result[k] = [0, 0]
        # 将同名文件的统计结果相加
        final_result[k][0] += v[0]  # 累加总函数数
        final_result[k][1] += v[1]  # 累加覆盖函数数

# 最后将所有的函数名称分别写入两个 .txt 文件中，每行一个函数名
# with open("/home/cvgroup/myz/wr/netsv/total_functions.txt", "w", encoding="utf-8") as f_total:
#     for name in all_total_function_names:
#         f_total.write(name + "\n")

with open("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/covered_api/resnet_covered_functions.txt", "w", encoding="utf-8") as f_covered:
    for name in all_covered_function_names:
        f_covered.write(name + "\n")




#---------------------------------------------------------
#   4) 查看最终结果 final_result
#   这时 final_result 已经累加了所有测试用例的结果
#---------------------------------------------------------
# print("=== 最终累加结果(final_result) ===")
# for py_file_path, (total_func_sum, covered_func_sum) in final_result.items():
#     print(f"{py_file_path} => 总函数数累计: {total_func_sum}, 覆盖函数数累计: {covered_func_sum}")

    
total_func_sum_all = 0
covered_func_sum_all = 0

for _, (total_func_sum, covered_func_sum) in final_result.items():
    total_func_sum_all += total_func_sum
    covered_func_sum_all += covered_func_sum

if total_func_sum_all == 0:
    print("没有统计到任何函数，总覆盖率无法计算。")
else:
    overall_coverage = covered_func_sum_all / total_func_sum_all

    print(f"===> 覆盖函数数量: {covered_func_sum_all}")
    print(f"===> 总数量: {total_func_sum_all}")

    print(f"===> 覆盖率: {overall_coverage:.6%}")  # 输出成百分比格式



# file_exists = os.path.isfile(csv_file_path)
# with open(csv_file_path, mode='a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     if not file_exists:
#         # 写入表头
#         writer.writerow(["Test Case","API coverage","covered_func_sum_all","total_func_sum_all"])
#     # 写入数据
#     writer.writerow([args.script_path, f"{overall_coverage:.6%}", f"{covered_func_sum_all}", f"{total_func_sum_all}"])
    # writer.writerow([f"{overall_coverage:.6%}s"])
# csv_file_path = '/home/cvgroup/myz/wr/netsv/common/log/ssimae-2025.2.20.6.4.44/filling_ssimae.csv'
# file_exists = os.path.isfile(csv_file_path)

# # 打开CSV文件，以读取模式和追加模式结合打开
# with open(csv_file_path, mode='r+', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     rows = list(reader)
# # 如果文件已经有表头，直接插入
#     if not file_exists or len(rows[0]) < 12: # 如果文件没有表头或者L列不存在
#         rows[0].append("coverage") # 在表头第L列插入coverage
# # 写入第二行第L列数据
#     if len(rows) > 1:
#         rows[1].append(f"{overall_coverage:.6%}") # 第二行插入overall_coverage值
#     else:
#         rows.append([f"{overall_coverage:.6%}"]) # 如果第二行不存在，则添加
# # 将修改后的数据重新写入文件
#     csvfile.seek(0) # 移动文件指针到开头
#     writer = csv.writer(csvfile)
#     writer.writerows(rows)
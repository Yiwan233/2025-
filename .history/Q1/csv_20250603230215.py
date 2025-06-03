# -*- coding: utf-8 -*-
"""
Excel转CSV转换工具
专为STR图谱数据设计，支持多种Excel格式转换为CSV

作者: 数学建模团队
日期: 2025-06-03
版本: V1.0
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings

# 忽略pandas警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('xlsx_to_csv_conversion.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ExcelToCSVConverter:
    """
    Excel转CSV转换器
    支持单文件转换、批量转换、多工作表处理等功能
    """
    
    def __init__(self):
        """初始化转换器"""
        self.supported_formats = ['.xlsx', '.xls', '.xlsm', '.xlsb']
        self.conversion_log = []
        
        logger.info("Excel转CSV转换器初始化完成")
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        验证文件是否为支持的Excel格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否为支持的格式
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return False
        
        if file_path.suffix.lower() not in self.supported_formats:
            logger.error(f"不支持的文件格式: {file_path.suffix}")
            return False
        
        return True
    
    def get_sheet_names(self, excel_file: Union[str, Path]) -> List[str]:
        """
        获取Excel文件中的所有工作表名称
        
        Args:
            excel_file: Excel文件路径
            
        Returns:
            List[str]: 工作表名称列表
        """
        try:
            with pd.ExcelFile(excel_file) as xls:
                return xls.sheet_names
        except Exception as e:
            logger.error(f"读取工作表名称失败: {e}")
            return []
    
    def convert_single_sheet(self, excel_file: Union[str, Path], 
                           sheet_name: str = None,
                           output_dir: Union[str, Path] = None,
                           encoding: str = 'utf-8-sig',
                           **kwargs) -> Optional[str]:
        """
        转换单个工作表为CSV
        
        Args:
            excel_file: Excel文件路径
            sheet_name: 工作表名称，None表示第一个工作表
            output_dir: 输出目录
            encoding: CSV编码格式
            **kwargs: pandas.read_excel的其他参数
            
        Returns:
            str: 输出CSV文件路径，失败返回None
        """
        excel_file = Path(excel_file)
        
        if not self.validate_file(excel_file):
            return None
        
        try:
            # 设置输出目录
            if output_dir is None:
                output_dir = excel_file.parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # 读取Excel
            logger.info(f"正在读取文件: {excel_file}")
            
            # 设置默认读取参数
            read_params = {
                'sheet_name': sheet_name,
                'engine': 'openpyxl' if excel_file.suffix == '.xlsx' else None,
                **kwargs
            }
            
            df = pd.read_excel(excel_file, **read_params)
            
            # 生成输出文件名
            if sheet_name:
                output_name = f"{excel_file.stem}_{sheet_name}.csv"
            else:
                output_name = f"{excel_file.stem}.csv"
            
            output_path = output_dir / output_name
            
            # 数据预处理
            df = self.preprocess_dataframe(df)
            
            # 保存为CSV
            df.to_csv(output_path, index=False, encoding=encoding)
            
            # 记录转换信息
            conversion_info = {
                'source_file': str(excel_file),
                'sheet_name': sheet_name or 'Sheet1',
                'output_file': str(output_path),
                'rows': len(df),
                'columns': len(df.columns),
                'status': 'success'
            }
            self.conversion_log.append(conversion_info)
            
            logger.info(f"转换成功: {excel_file} -> {output_path}")
            logger.info(f"数据维度: {len(df)} 行 × {len(df.columns)} 列")
            
            return str(output_path)
            
        except Exception as e:
            error_info = {
                'source_file': str(excel_file),
                'sheet_name': sheet_name,
                'error': str(e),
                'status': 'failed'
            }
            self.conversion_log.append(error_info)
            
            logger.error(f"转换失败: {excel_file} - {e}")
            return None
    
    def convert_all_sheets(self, excel_file: Union[str, Path],
                          output_dir: Union[str, Path] = None,
                          encoding: str = 'utf-8-sig',
                          **kwargs) -> List[str]:
        """
        转换Excel文件中的所有工作表
        
        Args:
            excel_file: Excel文件路径
            output_dir: 输出目录
            encoding: CSV编码格式
            **kwargs: pandas.read_excel的其他参数
            
        Returns:
            List[str]: 成功转换的CSV文件路径列表
        """
        excel_file = Path(excel_file)
        
        if not self.validate_file(excel_file):
            return []
        
        sheet_names = self.get_sheet_names(excel_file)
        if not sheet_names:
            logger.error(f"无法获取工作表: {excel_file}")
            return []
        
        logger.info(f"发现 {len(sheet_names)} 个工作表: {sheet_names}")
        
        converted_files = []
        for sheet_name in sheet_names:
            logger.info(f"正在转换工作表: {sheet_name}")
            output_file = self.convert_single_sheet(
                excel_file, sheet_name, output_dir, encoding, **kwargs
            )
            if output_file:
                converted_files.append(output_file)
        
        logger.info(f"共转换 {len(converted_files)}/{len(sheet_names)} 个工作表")
        return converted_files
    
    def batch_convert(self, input_dir: Union[str, Path],
                     output_dir: Union[str, Path] = None,
                     pattern: str = "*.xlsx",
                     all_sheets: bool = False,
                     encoding: str = 'utf-8-sig',
                     **kwargs) -> Dict[str, List[str]]:
        """
        批量转换目录中的Excel文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            pattern: 文件匹配模式
            all_sheets: 是否转换所有工作表
            encoding: CSV编码格式
            **kwargs: pandas.read_excel的其他参数
            
        Returns:
            Dict[str, List[str]]: 转换结果字典
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return {}
        
        # 查找Excel文件
        excel_files = []
        for ext in self.supported_formats:
            pattern_with_ext = pattern.replace('*', f'*{ext}')
            excel_files.extend(input_dir.glob(pattern_with_ext))
        
        if not excel_files:
            logger.warning(f"在目录 {input_dir} 中未找到Excel文件")
            return {}
        
        logger.info(f"找到 {len(excel_files)} 个Excel文件")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = input_dir / "csv_output"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 批量转换
        results = {}
        for excel_file in excel_files:
            logger.info(f"正在处理: {excel_file.name}")
            
            if all_sheets:
                converted_files = self.convert_all_sheets(
                    excel_file, output_dir, encoding, **kwargs
                )
            else:
                converted_file = self.convert_single_sheet(
                    excel_file, None, output_dir, encoding, **kwargs
                )
                converted_files = [converted_file] if converted_file else []
            
            results[str(excel_file)] = converted_files
        
        logger.info(f"批量转换完成，输出目录: {output_dir}")
        return results
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理（针对STR图谱数据优化）
        
        Args:
            df: 原始DataFrame
            
        Returns:
            pd.DataFrame: 预处理后的DataFrame
        """
        # 1. 处理列名
        df.columns = df.columns.astype(str)
        
        # 2. 移除完全空白的行和列
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # 3. 处理STR数据中的特殊值
        # 将明显的缺失值标记替换为NaN
        df = df.replace(['', ' ', 'N/A', 'n/a', 'NULL', 'null', '-'], np.nan)
        
        # 4. 处理数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # 将负数峰高设为0（STR数据中峰高不应为负）
            if 'height' in col.lower() or 'rfu' in col.lower():
                df[col] = df[col].clip(lower=0)
        
        # 5. 处理等位基因列
        for col in df.columns:
            if 'allele' in col.lower():
                # 标准化等位基因表示
                df[col] = df[col].astype(str).replace('nan', np.nan)
        
        return df
    
    def generate_conversion_report(self, output_file: str = "conversion_report.txt"):
        """
        生成转换报告
        
        Args:
            output_file: 报告文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Excel转CSV转换报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"转换时间: {pd.Timestamp.now()}\n")
                f.write(f"总转换任务数: {len(self.conversion_log)}\n\n")
                
                # 统计成功和失败的转换
                successful = [log for log in self.conversion_log if log.get('status') == 'success']
                failed = [log for log in self.conversion_log if log.get('status') == 'failed']
                
                f.write(f"成功转换: {len(successful)} 个\n")
                f.write(f"转换失败: {len(failed)} 个\n\n")
                
                # 详细转换记录
                f.write("详细转换记录:\n")
                f.write("-" * 30 + "\n\n")
                
                for i, log in enumerate(self.conversion_log, 1):
                    f.write(f"{i}. {log.get('source_file', 'Unknown')}\n")
                    if log.get('status') == 'success':
                        f.write(f"   状态: ✓ 成功\n")
                        f.write(f"   工作表: {log.get('sheet_name', 'N/A')}\n")
                        f.write(f"   输出: {log.get('output_file', 'N/A')}\n")
                        f.write(f"   数据: {log.get('rows', 'N/A')} 行 × {log.get('columns', 'N/A')} 列\n")
                    else:
                        f.write(f"   状态: ✗ 失败\n")
                        f.write(f"   错误: {log.get('error', 'N/A')}\n")
                    f.write("\n")
                
                # 文件大小统计
                if successful:
                    total_rows = sum(log.get('rows', 0) for log in successful)
                    total_cols = sum(log.get('columns', 0) for log in successful)
                    f.write(f"数据统计:\n")
                    f.write(f"  总行数: {total_rows:,}\n")
                    f.write(f"  总列数: {total_cols:,}\n")
                    f.write(f"  平均行数: {total_rows/len(successful):.0f}\n")
                    f.write(f"  平均列数: {total_cols/len(successful):.0f}\n")
            
            logger.info(f"转换报告已生成: {output_file}")
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")


def main():
    """
    命令行主函数
    """
    parser = argparse.ArgumentParser(
        description="Excel转CSV转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换单个文件
  python xlsx_to_csv.py -f data.xlsx
  
  # 转换文件的所有工作表
  python xlsx_to_csv.py -f data.xlsx --all-sheets
  
  # 批量转换目录中的所有Excel文件
  python xlsx_to_csv.py -d ./excel_files/ --batch
  
  # 指定输出目录和编码
  python xlsx_to_csv.py -f data.xlsx -o ./output/ --encoding utf-8
        """
    )
    
    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--file', type=str, help='单个Excel文件路径')
    input_group.add_argument('-d', '--directory', type=str, help='包含Excel文件的目录路径')
    
    # 输出参数
    parser.add_argument('-o', '--output', type=str, help='输出目录路径')
    parser.add_argument('--encoding', type=str, default='utf-8-sig', 
                       help='CSV文件编码 (默认: utf-8-sig)')
    
    # 转换选项
    parser.add_argument('--all-sheets', action='store_true', 
                       help='转换所有工作表（仅对单文件有效）')
    parser.add_argument('--batch', action='store_true', 
                       help='批量转换模式（仅对目录有效）')
    parser.add_argument('--pattern', type=str, default='*.xlsx', 
                       help='文件匹配模式 (默认: *.xlsx)')
    parser.add_argument('--sheet', type=str, 
                       help='指定要转换的工作表名称')
    
    # 其他选项
    parser.add_argument('--report', action='store_true', 
                       help='生成转换报告')
    parser.add_argument('--verbose', action='store_true', 
                       help='详细输出模式')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建转换器
    converter = ExcelToCSVConverter()
    
    try:
        if args.file:
            # 单文件转换
            logger.info(f"开始转换文件: {args.file}")
            
            if args.all_sheets:
                converted_files = converter.convert_all_sheets(
                    args.file, args.output, args.encoding
                )
                logger.info(f"转换完成，共生成 {len(converted_files)} 个CSV文件")
            else:
                converted_file = converter.convert_single_sheet(
                    args.file, args.sheet, args.output, args.encoding
                )
                if converted_file:
                    logger.info(f"转换完成: {converted_file}")
                else:
                    logger.error("转换失败")
                    return 1
        
        elif args.directory:
            # 目录转换
            logger.info(f"开始批量转换目录: {args.directory}")
            
            results = converter.batch_convert(
                args.directory, args.output, args.pattern, 
                args.batch, args.encoding
            )
            
            total_files = sum(len(files) for files in results.values())
            logger.info(f"批量转换完成，共生成 {total_files} 个CSV文件")
        
        # 生成报告
        if args.report:
            converter.generate_conversion_report()
        
        return 0
        
    except Exception as e:
        logger.error(f"程序执行错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import { Spin, Alert } from 'antd';
import { Link } from 'react-router-dom';
import { useRna } from '../context/RnaContext';

// 定义树节点类型
interface TreeNode {
  name: string;
  isPredicted?: boolean;
  children?: TreeNode[];
}

// 递归处理树数据，为每个节点添加样式
const processTreeData = (node: TreeNode): any => {
  const processed: any = {
    name: node.name,
    children: node.children?.map(child => processTreeData(child)),
  };

  // 根据 isPredicted 属性设置节点样式
  if (node.isPredicted) {
    // 预测结果：高亮显示（使用醒目的颜色和较大的节点）
    processed.itemStyle = {
      color: '#52c41a', // 绿色表示预测结果
      borderColor: '#389e0d',
      borderWidth: 2,
    };
    processed.label = {
      ...processed.label,
      color: '#52c41a',
      fontWeight: 'bold',
      fontSize: 14,
    };
    processed.symbolSize = 15; // 预测节点更大
  } else {
    // 未预测结果：灰色显示，但仍可见
    processed.itemStyle = {
      color: '#d9d9d9', // 灰色
      borderColor: '#bfbfbf',
      borderWidth: 1,
    };
    processed.label = {
      ...processed.label,
      color: '#8c8c8c',
      fontSize: 12,
    };
    processed.symbolSize = 10;
  }

  return processed;
};

const ClassificationViz: React.FC = () => {
  // 用于存储ECharts的配置对象
  const [options, setOptions] = useState({});
  // 用于控制加载状态，提供更好的用户体验
  const [loading, setLoading] = useState(true);
  // 用于存储请求过程中可能发生的错误
  const [error, setError] = useState<string | null>(null);

  const { rnaSequence } = useRna();

  // 使用useEffect在组件加载后执行数据获取操作
  useEffect(() => {
    if (!rnaSequence) {
      setLoading(false);
      setError("No RNA sequence provided.");
      return;
    }
    // 定义数据获取函数
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ sequence: rnaSequence }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const apiData = await response.json();

        // 从API数据中提取分类结果部分
        const chartData = apiData.classification;

        // 处理树数据，添加样式
        const processedData = processTreeData(chartData);

        // 构建ECharts的配置对象
        setOptions({
          tooltip: {
            trigger: 'item',
            triggerOn: 'mousemove',
            formatter: (params: any) => {
              const isPredicted = params.data.itemStyle?.color === '#52c41a';
              const status = isPredicted ? '✓ 预测结果' : '✗ 未预测';
              return `<strong>${params.name}</strong><br/>${status}`;
            }
          },
          series: [
            {
              type: 'tree',
              data: [processedData],
              // 从上到下的布局设置
              top: '5%',
              left: '10%',
              bottom: '5%',
              right: '10%',
              layout: 'orthogonal', // 正交布局
              orient: 'TB', // TB = Top to Bottom（从上到下）
              symbolSize: 10,
              initialTreeDepth: 2, // 默认展开两层
              label: {
                position: 'top',
                distance: 5,
                fontSize: 12,
              },
              leaves: {
                label: {
                  position: 'bottom',
                  distance: 5,
                }
              },
              emphasis: {
                focus: 'descendant'
              },
              expandAndCollapse: true,
              animationDuration: 550,
              animationDurationUpdate: 750
            }
          ]
        });

      } catch (e: any) {
        console.error("Failed to fetch or process data:", e);
        setError(`无法加载图表数据: ${e.message}`);
      } finally {
        // 无论成功或失败，都结束加载状态
        setLoading(false);
      }
    };

    fetchData();
  }, [rnaSequence]);

  if (!rnaSequence) {
    return (
      <Alert
        message="错误"
        description={
          <>
            请输入RNA序列. <Link to="/">返回主页</Link>
          </>
        }
        type="error"
        showIcon
      />
    );
  }

  // 根据加载和错误状态显示不同的UI
  if (loading) {
    return <Spin tip="正在加载图表..." size="large" style={{ display: 'block', marginTop: '50px' }} />;
  }

  if (error) {
    return <Alert message="错误" description={error} type="error" showIcon />;
  }

  // 数据加载成功后，渲染图表
  return (
    <div className="classification-viz-container">
      <ReactECharts
        option={options}
        style={{ height: '80vh', width: '100%', minWidth: '100%' }}
        notMerge={true}
        lazyUpdate={true}
      />
    </div>
  );
};

export default ClassificationViz;

import React, { useState, useEffect, useMemo } from 'react';
import { Spin, Alert, InputNumber, Button, Space, Card } from 'antd';
import { ArrowLeftOutlined, ArrowRightOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { useRna } from '../context/RnaContext';

// 定义权重和API数据的类型
interface Weight {
  index: number;
  type: string;
  score: number;
}
interface AttentionData {
  sequence: string;
  weights: Weight[];
}

// 定义莫兰迪配色方案
const baseColors: { [key: string]: string } = {
  'A': '#bcaaa4', // 柔和的灰玫瑰色 (腺嘌呤)
  'G': '#a5d6a7', // 柔和的鼠尾草绿 (鸟嘌呤)
  'C': '#90caf9', // 柔和的石板蓝 (胞嘧啶)
  'U': '#ffe082', // 柔和的沙黄色 (尿嘧啶)
  '-': '#eeeeee', // 用于填充字符的中性灰色
};

// 定义视图窗口的宽度（奇数以保证完美居中）
const VIEWPORT_WIDTH = 101;

const AttentionViz: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // 新增：存储从后端获取的完整数据
  const [sequence, setSequence] = useState('');
  const [allWeights, setAllWeights] = useState<Weight[]>([]);

  const [topX, setTopX] = useState<number>(3);
  const [currentIndex, setCurrentIndex] = useState(0);

  const { rnaSequence } = useRna();

  // 核心逻辑1：获取并存储完整数据
  useEffect(() => {
    if (!rnaSequence) {
      setLoading(false);
      setError("No RNA sequence provided.");
      return;
    }

    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch('http://localhost:5000/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sequence: rnaSequence }),
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const apiData = await response.json();
        const data: AttentionData = apiData.attention;

        setSequence(data.sequence);
        setAllWeights(data.weights); // 存储所有权重

      } catch (e: any) {
        setError(`无法加载数据: ${e.message}`);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [rnaSequence]);

  // 核心逻辑2：根据 topX 筛选要显示的权重 (使用 useMemo 优化性能)
  const displayWeights = useMemo(() => {
    if (!allWeights) return [];
    // 1. 前端负责排序和筛选
    return [...allWeights]
      .sort((a, b) => b.score - a.score)
      .slice(0, topX);
  }, [allWeights, topX]);

  // 当 topX 变化时，重置当前查看的索引
  useEffect(() => {
    setCurrentIndex(0);
  }, [topX]);

  const handlePrev = () => setCurrentIndex(i => (i > 0 ? i - 1 : i));
  const handleNext = () => setCurrentIndex(i => (i < displayWeights.length - 1 ? i + 1 : i));

  // 核心逻辑3：生成居中且带填充的序列视图 - 使用彩色方块
  const renderSequenceViewport = () => {
    if (displayWeights.length === 0) return null;

    const currentHighlight = displayWeights[currentIndex];
    const centerIndex = currentHighlight.index;

    const halfWidth = Math.floor(VIEWPORT_WIDTH / 2);
    const startIndex = centerIndex - halfWidth;
    const endIndex = centerIndex + halfWidth;

    const viewportElements = [];

    for (let i = startIndex; i <= endIndex; i++) {
      const isHighlighted = i === centerIndex;
      const isOutOfBounds = i < 0 || i >= sequence.length;

      let base = isOutOfBounds ? '-' : sequence[i];
      const bgColor = baseColors[base] || baseColors['-'];

      if (isHighlighted) {
        viewportElements.push(
          <div key={i} className="highlight-container-block">
            <div className="annotation-label-block">{currentHighlight.type} ({currentHighlight.score.toFixed(2)}) (Index:{currentHighlight.index})</div>
            <div
              className="sequence-block highlighted-block"
              style={{ backgroundColor: bgColor }}
            >
              {base}
            </div>
          </div>
        );
      } else {
        viewportElements.push(
          <div
            key={i}
            className="sequence-block"
            style={{ backgroundColor: bgColor }}
          >
            {base}
          </div>
        );
      }
    }
    return viewportElements;
  };

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

  if (loading) return <Spin tip="正在加载序列数据..." size="large" style={{ display: 'block', marginTop: '50px' }} />;
  if (error) return <Alert message="错误" description={error} type="error" showIcon />;

  return (
    <div className="space-wrapper">
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        {/* 控制区 */}
        <Card title="可视化控制">
          <Space wrap>
            <span>显示 Top</span>
            <InputNumber min={1} max={1001} value={topX} onChange={(value) => setTopX(value || 1)} />
            <span>个修饰位点</span>
            <Button icon={<ArrowLeftOutlined />} onClick={handlePrev} disabled={currentIndex === 0}>
              上一个
            </Button>
            <span>
              当前查看: 第 {displayWeights.length > 0 ? currentIndex + 1 : 0} / {displayWeights.length} 个
            </span>
            <Button icon={<ArrowRightOutlined />} onClick={handleNext} disabled={currentIndex >= displayWeights.length - 1}>
              下一个
            </Button>
          </Space>
        </Card>

        <Card className="card-wrapper">
          {/* 序列视图 - 彩色方块布局 */}
          <div className="sequence-viewport-container-block">
            {renderSequenceViewport()}
          </div>
        </Card>
      </Space>
    </div>
  );
};

export default AttentionViz;

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
  originalScore?: number;  // 保存原始分数
  normalizedScore?: number;  // 保存归一化分数
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

  const viewportRef = React.useRef<HTMLDivElement>(null);

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

        console.log('API Response:', apiData);
        console.log('Attention Data:', data);
        console.log('Weights:', data.weights);

        setSequence(data.sequence);
        setAllWeights(data.weights); // 存储所有权重

      } catch (e: any) {
        console.error('Error fetching data:', e);
        setError(`无法加载数据: ${e.message}`);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [rnaSequence]);

  // 核心逻辑2：根据 topX 筛选要显示的权重，并进行归一化 (使用 useMemo 优化性能)
  const displayWeights = useMemo(() => {
    if (!allWeights || allWeights.length === 0) return [];

    // Step 1: 按核苷酸组（type）分组
    const groups: { [key: string]: Weight[] } = {};
    allWeights.forEach(weight => {
      if (!groups[weight.type]) {
        groups[weight.type] = [];
      }
      groups[weight.type].push(weight);
    });

    // Step 2: 对每个组内的分数进行归一化
    const normalizedWeights: Weight[] = [];
    Object.keys(groups).forEach(type => {
      const groupWeights = groups[type];

      // 找到该组内的最大和最小分数
      const scores = groupWeights.map(w => w.score);
      const maxScore = Math.max(...scores);
      const minScore = Math.min(...scores);
      const scoreRange = maxScore - minScore;

      // 归一化到 [0, 1] 范围
      // 如果所有分数相同（range = 0），则都设为 1
      groupWeights.forEach(weight => {
        let normalizedScore: number;
        if (scoreRange === 0) {
          normalizedScore = 1.0;
        } else {
          normalizedScore = (weight.score - minScore) / scoreRange;
        }

        normalizedWeights.push({
          ...weight,
          originalScore: weight.score,  // 保存原始分数
          normalizedScore: normalizedScore,  // 保存归一化分数
          score: normalizedScore  // 使用归一化分数用于排序和显示
        });
      });
    });

    // Step 3: 按归一化后的分数排序并筛选 topX
    return normalizedWeights
      .sort((a, b) => b.score - a.score)
      .slice(0, topX);
  }, [allWeights, topX]);

  // 当 topX 变化时，重置当前查看的索引
  useEffect(() => {
    setCurrentIndex(0);
  }, [topX]);

  // 当 currentIndex 变化时，滚动到高亮元素
  useEffect(() => {
    if (viewportRef.current && displayWeights.length > 0) {
      const highlightedElement = viewportRef.current.querySelector('.highlight-container-block');
      if (highlightedElement) {
        highlightedElement.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest',
          inline: 'center'
        });
      }
    }
  }, [currentIndex, displayWeights.length]);

  const handlePrev = () => setCurrentIndex(i => (i > 0 ? i - 1 : i));
  const handleNext = () => setCurrentIndex(i => (i < displayWeights.length - 1 ? i + 1 : i));

  // 核心逻辑3：生成居中且带填充的序列视图 - 使用彩色方块
  const renderSequenceViewport = () => {
    if (displayWeights.length === 0) {
      return (
        <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
          <p>没有检测到显著修饰位点</p>
          <p style={{ fontSize: '14px', marginTop: '10px' }}>
            当前序列长度: {sequence.length} | Top设置: {topX} | 权重数据总数: {allWeights.length}
          </p>
        </div>
      );
    }

    console.log('Rendering viewport with displayWeights:', displayWeights);
    console.log('Current index:', currentIndex);
    console.log('Current highlight:', displayWeights[currentIndex]);

    const currentHighlight = displayWeights[currentIndex];
    const centerIndex = currentHighlight.index;

    const halfWidth = Math.floor(VIEWPORT_WIDTH / 2);
    const startIndex = centerIndex - halfWidth;
    const endIndex = centerIndex + halfWidth;

    console.log('Viewport range:', { startIndex, endIndex, centerIndex, sequenceLength: sequence.length });

    const viewportElements = [];

    for (let i = startIndex; i <= endIndex; i++) {
      const isHighlighted = i === centerIndex;
      const isOutOfBounds = i < 0 || i >= sequence.length;

      let base = isOutOfBounds ? '-' : sequence[i];
      const bgColor = baseColors[base] || baseColors['-'];

      if (isHighlighted) {
        console.log('Rendering highlighted element at index:', i, 'base:', base);
        viewportElements.push(
          <div key={i} className="highlight-container-block">
            <div className="annotation-label-block">
              {currentHighlight.type} (
                {/* 归一化: {currentHighlight.score.toFixed(3)} | */}
                 原始: {currentHighlight.originalScore?.toFixed(6) ?? 'N/A'}) (Index:{currentHighlight.index})
            </div>
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
          {/* 调试信息 */}
          <div style={{ marginBottom: '10px', padding: '10px', background: '#f0f0f0', borderRadius: '4px' }}>
            <strong>渲染调试信息:</strong>
            <p>序列长度: {sequence.length}</p>
            <p>当前高亮: {displayWeights.length > 0 ? `位置 ${displayWeights[currentIndex].index}, 类型 ${displayWeights[currentIndex].type}` : '无'}</p>
            <p>视口元素数量: {displayWeights.length > 0 ? VIEWPORT_WIDTH : 0}</p>
          </div>

          {/* 序列视图 - 彩色方块布局 */}
          <div ref={viewportRef} className="sequence-viewport-container-block">
            {renderSequenceViewport()}
          </div>
        </Card>

        {/* 显示统计信息 */}
        <Card title="修饰位点统计" style={{ marginTop: '20px' }}>
          <p>原始序列长度: <strong>{sequence.length}</strong></p>
          <p>检测到的权重数据总数: <strong>{allWeights.length}</strong></p>
          <p>当前显示 Top: <strong>{topX}</strong></p>
          <p>实际显示: <strong>{displayWeights.length}</strong> 个修饰位点</p>

          {displayWeights.length > 0 && (
            <div style={{ marginTop: '15px' }}>
              <strong>当前显示的修饰位点列表 (同组核苷酸归一化):</strong>
              <ul style={{ marginTop: '10px', paddingLeft: '20px' }}>
                {displayWeights.map((weight, idx) => (
                  <li key={idx}>
                    <strong>{idx + 1}. {weight.type}</strong> - 位置: {weight.index},
                    {/* 归一化得分: {weight.score.toFixed(4)} | */}
                    原始得分: {weight.originalScore?.toFixed(6) ?? 'N/A'}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      </Space>
    </div>
  );
};

export default AttentionViz;

import React, { useState, useEffect } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import { Spin, Alert } from 'antd';
import { Link } from 'react-router-dom';
import { useRna } from '../context/RnaContext';

interface Node {
  id: string;
  label?: string;
  data?: {
    index: number;
    type: string;
  };
  x?: number;
  y?: number;
  z?: number;
}

interface Link {
  source: string | Node;
  target: string | Node;
}

interface GraphData {
  nodes: Node[];
  edges: Link[];
}

interface TooltipState {
  visible: boolean;
  content: string;
  x: number;
  y: number;
}

const GcnViz: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [gcnData, setGcnData] = useState<GraphData | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false,
    content: '',
    x: 0,
    y: 0,
  });
  const graphRef = React.useRef<any>(null);

  const { rnaSequence } = useRna();

  useEffect(() => {
    const fetchData = async () => {
      if (!rnaSequence) {
        setLoading(false);
        setError("No RNA sequence provided.");
        return;
      }

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
        const graphData: GraphData = apiData.gcn;

        // Add labels to nodes for display
        if (graphData.nodes) {
          graphData.nodes.forEach((node: any) => {
            node.label = node.id;
          });
        }

        // Process links: convert string source/target to node references
        if (graphData.edges && graphData.nodes) {
          const nodeMap = new Map<string, Node>();
          graphData.nodes.forEach((node: Node) => {
            nodeMap.set(node.id, node);
          });

          graphData.edges.forEach((link: any) => {
            if (typeof link.source === 'string') {
              link.source = nodeMap.get(link.source);
            }
            if (typeof link.target === 'string') {
              link.target = nodeMap.get(link.target);
            }
          });
        }

        setGcnData(graphData);
      } catch (e: any) {
        setError(`无法加载图数据: ${e.message}`);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [rnaSequence]);

  // Handle node hover (PC端悬停交互)
  const handleNodeHover = (node: Node | null) => {
    if (node && graphRef.current) {
      try {
        // 使用 graphRef 获取节点的屏幕坐标
        const camera = graphRef.current.camera();
        const Vector3 = graphRef.current.Three?.Vector3 || THREE.Vector3;
        const pos = new Vector3(node.x || 0, node.y || 0, node.z || 0);
        pos.project(camera);

        const x = (pos.x * 0.5 + 0.5) * window.innerWidth;
        const y = (-(pos.y * 0.5) + 0.5) * window.innerHeight;

        setTooltip({
          visible: true,
          content: `节点: ${node.id}${node.data ? `\n索引: ${node.data.index}\n类型: ${node.data.type}` : ''}`,
          x: x,
          y: y + 60, // 在节点下方显示
        });
      } catch (e) {
        // 如果计算失败，使用鼠标位置
        const evt = window.event as MouseEvent;
        if (evt) {
          setTooltip({
            visible: true,
            content: `节点: ${node.id}${node.data ? `\n索引: ${node.data.index}\n类型: ${node.data.type}` : ''}`,
            x: evt.clientX,
            y: evt.clientY + 60,
          });
        }
      }
    } else {
      setTooltip(prev => ({ ...prev, visible: false }));
    }
  };

  // Handle node click (移动端点击交互)
  const handleNodeClick = (node: Node) => {
    if (graphRef.current) {
      try {
        const camera = graphRef.current.camera();
        const Vector3 = graphRef.current.Three?.Vector3 || THREE.Vector3;
        const pos = new Vector3(node.x || 0, node.y || 0, node.z || 0);
        pos.project(camera);

        const x = (pos.x * 0.5 + 0.5) * window.innerWidth;
        const y = (-(pos.y * 0.5) + 0.5) * window.innerHeight;

        setTooltip({
          visible: true,
          content: `节点: ${node.id}${node.data ? `\n索引: ${node.data.index}\n类型: ${node.data.type}` : ''}`,
          x: x,
          y: y + 60,
        });
      } catch (e) {
        // 如果计算失败，使用屏幕中心
        setTooltip({
          visible: true,
          content: `节点: ${node.id}${node.data ? `\n索引: ${node.data.index}\n类型: ${node.data.type}` : ''}`,
          x: window.innerWidth / 2,
          y: window.innerHeight / 2,
        });
      }
    }
  };

  // Handle background click to close tooltip (移动端关闭提示框)
  const handleBackgroundClick = () => {
    setTooltip(prev => ({ ...prev, visible: false }));
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

  return (
    <div className="gcn-viz-container" style={{ height: '80vh', position: 'relative', background: '#F5F5DC' }}>
      {loading && (
        <Spin
          tip="正在加载图数据..."
          size="large"
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 100,
          }}
        />
      )}
      {error && (
        <Alert
          message="错误"
          description={error}
          type="error"
          showIcon
          style={{ position: 'absolute', zIndex: 100 }}
        />
      )}

      {gcnData && (
        <ForceGraph3D
          ref={graphRef}
          graphData={{
            nodes: gcnData.nodes,
            links: gcnData.edges,
          }}
          nodeLabel="label"
          nodeColor="#5B8FF9"
          nodeRelSize={6}
          linkColor="#000000"
          linkWidth={2}
          backgroundColor="#F5F5DC"
          onNodeHover={handleNodeHover}
          onNodeClick={handleNodeClick}
          onBackgroundClick={handleBackgroundClick}
          enableNodeDrag={true}
          cooldownTicks={200}
        />
      )}

      {/* Tooltip 组件 */}
      {tooltip.visible && (
        <div
          style={{
            position: 'fixed',
            left: `${tooltip.x + 15}px`,
            top: `${tooltip.y + 15}px`,
            backgroundColor: 'rgba(0, 21, 41, 0.9)',
            color: '#fff',
            padding: '12px 16px',
            borderRadius: '6px',
            fontSize: '14px',
            whiteSpace: 'pre-line',
            zIndex: 1000,
            pointerEvents: 'none',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
            maxWidth: '250px',
          }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
};

export default GcnViz;

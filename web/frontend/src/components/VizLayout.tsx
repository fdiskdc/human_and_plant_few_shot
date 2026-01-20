import React, { useState, useEffect } from 'react';
import { Layout, Menu } from 'antd';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';

const { Content, Footer, Sider } = Layout;

const DESKTOP_BREAKPOINT = 768;

const VizLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isDesktop, setIsDesktop] = useState(window.innerWidth > DESKTOP_BREAKPOINT);
  const [siderCollapsed, setSiderCollapsed] = useState(false);

  const currentKey = location.pathname;

  const handleMenuClick = (e: any) => {
    navigate(e.key);
  };

  useEffect(() => {
    const handleResize = () => {
      const desktop = window.innerWidth > DESKTOP_BREAKPOINT;
      setIsDesktop(desktop);
      // Auto-collapse sidebar on mobile
      if (!desktop) {
        setSiderCollapsed(true);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const menuItems = [
    { key: '/', label: '返回主页' },
    { key: '/classification', label: '分类结果' },
    { key: '/attention', label: '注意力权重' },
    { key: '/gcn', label: 'GCN图结构' },
  ];

  // Desktop layout: sidebar on the left
  if (isDesktop) {
    return (
      <Layout style={{ minHeight: '100vh' }}>
        <Sider
          collapsible
          collapsed={siderCollapsed}
          onCollapse={setSiderCollapsed}
          theme="dark"
          width={200}
          style={{
            overflow: 'auto',
            height: '100vh',
            position: 'fixed',
            left: 0,
            top: 0,
            bottom: 0,
          }}
        >
          <div style={{
            height: 32,
            margin: 16,
            color: '#fff',
            fontWeight: 'bold',
            textAlign: 'center',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
          }}>
            {siderCollapsed ? 'RNA' : 'RNA 可视化'}
          </div>
          <Menu
            onClick={handleMenuClick}
            selectedKeys={[currentKey]}
            mode="inline"
            theme="dark"
            items={menuItems}
            inlineIndent={16}
          />
        </Sider>
        <Layout style={{ marginLeft: siderCollapsed ? 80 : 200 }}>
          <Content style={{
            padding: '20px',
            overflow: 'auto',
            minHeight: '100vh',
          }}>
            <Outlet />
          </Content>
        </Layout>
      </Layout>
    );
  }

  // Mobile layout: menu at the bottom
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Content style={{
        padding: '20px',
        overflow: 'auto',
        paddingBottom: 60, // Space for footer
      }}>
        <Outlet />
      </Content>
      <Footer style={{ padding: 0, position: 'fixed', bottom: 0, left: 0, right: 0, zIndex: 10 }}>
        <Menu
          onClick={handleMenuClick}
          selectedKeys={[currentKey]}
          mode="horizontal"
          theme="dark"
          style={{ justifyContent: 'center' }}
          items={menuItems}
        />
      </Footer>
    </Layout>
  );
};

export default VizLayout;

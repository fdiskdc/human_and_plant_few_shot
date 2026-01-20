import React, { useState } from 'react';
import { Input, Select, Button, Card } from 'antd';
import { useNavigate } from 'react-router-dom';
import { useRna } from '../context/RnaContext';

const { TextArea } = Input;
const { Option } = Select;

const MainPage: React.FC = () => {
    const [localRnaSequence, setLocalRnaSequence] = useState("AUGCCGUACGAUCGACGAUCGUACGUACGUACGUAUCGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACG");
    const [localServer, setLocalServer] = useState('server1');
    const { setRnaSequence, setServer } = useRna();
    const navigate = useNavigate();

    const handleSubmit = () => {
        setRnaSequence(localRnaSequence);
        setServer(localServer);
        navigate('/classification');
    };

    return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh' }}>
            <Card title="RNA 序列分析" style={{ width: 800 }}>
                <TextArea
                    rows={6}
                    value={localRnaSequence}
                    onChange={(e) => setLocalRnaSequence(e.target.value)}
                    placeholder="输入 RNA 序列"
                />
                <div style={{ margin: '20px 0' }}>
                    <Select value={localServer} onChange={(value) => setLocalServer(value)} style={{ width: '100%' }}>
                        <Option value="server1">服务器 1</Option>
                        <Option value="server2">服务器 2</Option>
                        <Option value="server3">服务器 3</Option>
                    </Select>
                </div>
                <Button type="primary" onClick={handleSubmit} block>
                    提交
                </Button>
            </Card>
        </div>
    );
};

export default MainPage;

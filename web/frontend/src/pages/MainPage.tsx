/*
 * @Author: Chao Deng && chaodeng987@outlook.com
 * @Date: 2026-01-20 08:39:31
 * @LastEditors: Chao Deng && chaodeng987@outlook.com
 * @LastEditTime: 2026-01-20 11:07:03
 * @FilePath: /rgcnformer_sum/web/frontend/src/pages/MainPage.tsx
 * @Description: 
 * 那只是一场游戏一场梦
 *  
 * https://orcid.org/0009-0009-8520-1656
 * DOI: 10.3390/app15158626
 * DOI: 10.3390/rs17142354
 * Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
 */
import React, { useState } from 'react';
import { Input, Select, Button, Card } from 'antd';
import { useNavigate } from 'react-router-dom';
import { useRna } from '../context/RnaContext';

const { TextArea } = Input;
const { Option } = Select;

const MainPage: React.FC = () => {
    const [localRnaSequence, setLocalRnaSequence] = useState("TCAGGAGTTCGAGACCAGCCTGATCAACATGACGAAACCCTATCTCTACTAAAAATACAAAAATTAGCCGGGCGTGGTGGCATGCGCCTGTAGTCTCAGCTACTTGGGAGGCTGAAGCAGGAGAATCGTTTGAACCCAGGAGGCAGAGGTTGCAGTGAGCCGAGATCGTGCCACTGCACTCCAGCCTGGGTGACACAGCGAGACTCTGTCTCAAAAAAATAAAAATAAAAAAATAAATAAATAACCTTTAATTTAGTGAGACTTCATATAGAATTGTTTTAATGTTTAATATAGACCATTTGTTTTAGGTGAATTTAACAATTTCATACTGTGATTAAGATTAATTTCTTTTTCTGACTTCTACCAGAAAGCAGGAATTATGTTTCAAATGGACAATCATTTACCAAACCTTGTTAATCTGAATGAAGATCCACAACTATCTGAGATGCTGCTATATATGATAAAAGAAGGAACAACTACAGTTGGAAAGTATAAACCAAACTCAAGCCATGATATTCAGTTATCTGGGGTGCTGATTGCTGATGATCATTGGTATGTTAATCCTCTAAAAAAAAAGAAAAGGCACCTGTTCTATATCTTGATAACATGTGGTTTCCTTCATATGGCATATTCGTTGATACTGATCGTTTGGTAGAATTCTTCAAACCCATTGTTTAGTCAGGAAAAACATACATTCTGAGTGTGTTATAAGGATGATAGGTCAGTTACTCTCAATATAAAGTACAGTGTAATGCTCTCTCTGTTTTTGTTTTGGCATACTTGATCTGTTGATTGAAGAATAATTTATTTTCTTGCAATTATAATGATGCACATGCAAGTAAACTATCTATCTTACATAACAGAATTTTTGGTTGGATTGACCAATTTAAAAATGTTACTTTATGTGAATTTTGTTCATATGAATGGAATACTTGTATATATTGTTGGAATGATAGCGTATGTAAACTTTTTTGACTCTGCATTGTGTTTCCAAGATTTGT");
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

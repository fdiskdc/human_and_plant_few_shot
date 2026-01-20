import React, { createContext, useState, useContext, ReactNode } from 'react';

interface RnaContextType {
    rnaSequence: string;
    setRnaSequence: (sequence: string) => void;
    server: string;
    setServer: (server: string) => void;
}

const RnaContext = createContext<RnaContextType | undefined>(undefined);

export const RnaProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [rnaSequence, setRnaSequence] = useState('');
    const [server, setServer] = useState('');

    return (
        <RnaContext.Provider value={{ rnaSequence, setRnaSequence, server, setServer }}>
            {children}
        </RnaContext.Provider>
    );
};

export const useRna = () => {
    const context = useContext(RnaContext);
    if (!context) {
        throw new Error('useRna must be used within a RnaProvider');
    }
    return context;
};

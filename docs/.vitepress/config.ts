import { defineConfig } from 'vitepress'

// VitePress i18n configuration
// Each locale has independent sidebar / nav / themeConfig.
// Root (/) is treated as Chinese; /en/ is English.
export default defineConfig({
  title: 'RGCNFormer 教程 / RGCNFormer Tutorial',
  description: 'RGCNFormer / mRModN RNA 修饰位点分类系统双语教程 / Bilingual tutorial for the RGCNFormer / mRModN RNA modification classification system',
  cleanUrls: true,
  // Locale-relative links like /guide/overview resolve differently per locale.
  // VitePress cannot disambiguate these at build time, so we ignore the warning.
  ignoreDeadLinks: true,
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      title: 'RGCNFormer 教程',
      description: '基于 mRModN 架构的 RNA 修饰位点分类系统教程',
      themeConfig: {
        nav: [
          { text: '首页', link: '/' },
          { text: '快速开始', link: '/guide/quickstart' },
          { text: '模型详解', link: '/guide/model' },
          { text: 'GitHub', link: 'https://github.com/' }
        ],
        sidebar: [
          {
            text: '概述',
            items: [
              { text: '项目概述', link: '/guide/overview' }
            ]
          },
          {
            text: '算法',
            items: [
              { text: '算法架构', link: '/guide/architecture' }
            ]
          },
          {
            text: '实践',
            items: [
              { text: '快速开始', link: '/guide/quickstart' },
              { text: '模型详解', link: '/guide/model' }
            ]
          },
          {
            text: '数据',
            items: [
              { text: '数据集', link: '/guide/dataset' }
            ]
          },
          {
            text: '训练与推理',
            items: [
              { text: '训练流程', link: '/guide/training' },
              { text: '推理流程', link: '/guide/inference' }
            ]
          },
          {
            text: '分析与可视化',
            items: [
              { text: '可视化', link: '/guide/visualization' },
              { text: '消融实验', link: '/guide/ablation' },
              { text: '少/零样本分析', link: '/guide/fewshot' }
            ]
          },
          {
            text: '附录',
            items: [
              { text: '附录', link: '/guide/appendix' }
            ]
          }
        ],
        socialLinks: [
          { icon: 'github', link: 'https://github.com/' }
        ],
        search: {
          provider: 'local'
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      title: 'RGCNFormer Tutorial',
      description: 'Tutorial for the RGCNFormer / mRModN RNA modification classification system',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Quickstart', link: '/en/guide/quickstart' },
          { text: 'Model', link: '/en/guide/model' },
          { text: 'GitHub', link: 'https://github.com/' }
        ],
        sidebar: [
          {
            text: 'Overview',
            items: [
              { text: 'Project Overview', link: '/en/guide/overview' }
            ]
          },
          {
            text: 'Algorithm',
            items: [
              { text: 'Algorithm Architecture', link: '/en/guide/architecture' }
            ]
          },
          {
            text: 'Practice',
            items: [
              { text: 'Quickstart', link: '/en/guide/quickstart' },
              { text: 'Model Details', link: '/en/guide/model' }
            ]
          },
          {
            text: 'Data',
            items: [
              { text: 'Datasets', link: '/en/guide/dataset' }
            ]
          },
          {
            text: 'Training & Inference',
            items: [
              { text: 'Training Pipeline', link: '/en/guide/training' },
              { text: 'Inference Pipeline', link: '/en/guide/inference' }
            ]
          },
          {
            text: 'Analysis & Visualization',
            items: [
              { text: 'Visualization', link: '/en/guide/visualization' },
              { text: 'Ablation Studies', link: '/en/guide/ablation' },
              { text: 'Few/Zero-shot Analysis', link: '/en/guide/fewshot' }
            ]
          },
          {
            text: 'Appendix',
            items: [
              { text: 'Appendix', link: '/en/guide/appendix' }
            ]
          }
        ],
        socialLinks: [
          { icon: 'github', link: 'https://github.com/' }
        ],
        search: {
          provider: 'local'
        }
      }
    }
  }
})

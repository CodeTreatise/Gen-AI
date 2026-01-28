// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import AstroPWA from '@vite-pwa/astro';

// https://astro.build/config
export default defineConfig({
  site: 'https://codetreatise.github.io',
  base: '/Gen-AI',
  integrations: [
    mermaid(),
    starlight({
      title: 'AI/ML Web Integration',
      description: 'A comprehensive guide to integrating AI and ML into web applications',
      logo: {
        light: './src/assets/logo-light.svg',
        dark: './src/assets/logo-dark.svg',
        replacesTitle: false,
      },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/CodeTreatise/Gen-AI' },
      ],
      customCss: ['./src/styles/custom.css'],
      head: [
        {
          tag: 'link',
          attrs: {
            rel: 'manifest',
            href: '/Gen-AI/manifest.json',
          },
        },
        {
          tag: 'meta',
          attrs: {
            name: 'theme-color',
            content: '#6366f1',
          },
        },
        {
          tag: 'meta',
          attrs: {
            name: 'apple-mobile-web-app-capable',
            content: 'yes',
          },
        },
        {
          tag: 'meta',
          attrs: {
            name: 'apple-mobile-web-app-status-bar-style',
            content: 'black-translucent',
          },
        },
        {
          tag: 'link',
          attrs: {
            rel: 'apple-touch-icon',
            href: '/Gen-AI/icons/icon-192x192.png',
          },
        },
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Introduction', slug: 'introduction' },
          ],
        },
        {
          label: '01. Web Development Fundamentals',
          collapsed: true,
          autogenerate: { directory: '01-web-development-fundamentals' },
        },
        {
          label: '02. Python for AI Development',
          collapsed: true,
          autogenerate: { directory: '02-python-for-ai-development' },
        },
        {
          label: '03. AI & LLM Fundamentals',
          collapsed: true,
          autogenerate: { directory: '03-ai-llm-fundamentals' },
        },
        {
          label: '04. AI API Integration',
          collapsed: true,
          autogenerate: { directory: '04-ai-api-integration' },
        },
        {
          label: '05. Conversational Interfaces',
          collapsed: true,
          autogenerate: { directory: '05-building-conversational-interfaces' },
        },
        {
          label: '06. Prompt Engineering',
          collapsed: true,
          autogenerate: { directory: '06-prompt-engineering' },
        },
        {
          label: '07. Embeddings & Vector Search',
          collapsed: true,
          autogenerate: { directory: '07-embeddings-vector-search' },
        },
        {
          label: '08. LangChain & LlamaIndex',
          collapsed: true,
          autogenerate: { directory: '08-langchain-llamaindex-mastery' },
        },
        {
          label: '09. RAG',
          collapsed: true,
          autogenerate: { directory: '09-rag-retrieval-augmented-generation' },
        },
        {
          label: '10. Function Calling & Tools',
          collapsed: true,
          autogenerate: { directory: '10-function-calling-tool-use' },
        },
        {
          label: '11. AI Agents',
          collapsed: true,
          autogenerate: { directory: '11-ai-agents' },
        },
        {
          label: '12. Multi-Agent Systems',
          collapsed: true,
          autogenerate: { directory: '12-multi-agent-systems' },
        },
        {
          label: '13. Image & Multimodal AI',
          collapsed: true,
          autogenerate: { directory: '13-image-multimodal-ai' },
        },
        {
          label: '14. Audio & Voice AI',
          collapsed: true,
          autogenerate: { directory: '14-audio-voice-ai' },
        },
        {
          label: '15. Client-Side ML',
          collapsed: true,
          autogenerate: { directory: '15-client-side-machine-learning' },
        },
        {
          label: '16. Production & Optimization',
          collapsed: true,
          autogenerate: { directory: '16-production-optimization' },
        },
        {
          label: '17. Cloud AI Platforms',
          collapsed: true,
          autogenerate: { directory: '17-cloud-ai-platforms' },
        },
        {
          label: '18. Security, Privacy & Ethics',
          collapsed: true,
          autogenerate: { directory: '18-security-privacy-ethics' },
        },
        {
          label: '19. Testing AI Applications',
          collapsed: true,
          autogenerate: { directory: '19-testing-ai-applications' },
        },
        {
          label: '20. AI UX & Design',
          collapsed: true,
          autogenerate: { directory: '20-ai-ux-conversational-design' },
        },
        {
          label: '21. AI-Powered Search',
          collapsed: true,
          autogenerate: { directory: '21-ai-powered-search-systems' },
        },
        {
          label: '22. Workflow Automation',
          collapsed: true,
          autogenerate: { directory: '22-ai-workflow-automation' },
        },
        {
          label: '23. Observability & Monitoring',
          collapsed: true,
          autogenerate: { directory: '23-ai-observability-monitoring-tools' },
        },
        {
          label: '24. ML Frameworks',
          collapsed: true,
          autogenerate: { directory: '24-machine-learning-frameworks' },
        },
        {
          label: '25. Fine-Tuning & Custom Models',
          collapsed: true,
          autogenerate: { directory: '25-fine-tuning-custom-models' },
        },
      ],
    }),
    AstroPWA({
      mode: 'production',
      base: '/Gen-AI/',
      scope: '/Gen-AI/',
      includeAssets: ['favicon.svg', 'icons/*.png'],
      registerType: 'autoUpdate',
      manifest: false, // Using external manifest.json
      workbox: {
        navigateFallback: '/Gen-AI/',
        globPatterns: ['**/*.{css,js,html,svg,png,ico,txt,woff,woff2}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/fonts\.googleapis\.com\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'google-fonts-cache',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365, // 1 year
              },
              cacheableResponse: {
                statuses: [0, 200],
              },
            },
          },
          {
            urlPattern: /^https:\/\/fonts\.gstatic\.com\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'gstatic-fonts-cache',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365, // 1 year
              },
              cacheableResponse: {
                statuses: [0, 200],
              },
            },
          },
          {
            urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp)$/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'images-cache',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 60 * 60 * 24 * 30, // 30 days
              },
            },
          },
          {
            urlPattern: /\.(?:js|css)$/i,
            handler: 'StaleWhileRevalidate',
            options: {
              cacheName: 'static-resources',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 60 * 60 * 24 * 7, // 7 days
              },
            },
          },
        ],
      },
      devOptions: {
        enabled: false,
      },
    }),
  ],
});

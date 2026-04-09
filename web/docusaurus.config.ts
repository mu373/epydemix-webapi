import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'epydemix Web API',
  tagline: 'REST API for epidemic simulations',
  favicon: undefined,

  url: 'https://epydemix-webapi.vercel.app',
  baseUrl: '/',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/mu373/epydemix-webapi/blob/main/',
          sidebarCollapsible: false,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.scss',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    'docusaurus-plugin-sass',
    [
      '@scalar/docusaurus',
      {
        label: 'API Reference',
        route: '/api-reference',
        showNavLink: false,
        configuration: {
          url: '/openapi.json',
          servers: [
            {url: 'https://epyscenario-api.isi.it', description: 'Production'},
            {url: 'http://localhost:8000', description: 'Local'},
          ],
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'epydemix Web API',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/api-reference',
          label: 'API Reference',
          position: 'left',
        },
        {
          href: 'https://github.com/mu373/epydemix-webapi',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Introduction', to: '/docs'},
            {label: 'API Reference', to: '/api-reference'},
          ],
        },
        {
          title: 'Links',
          items: [
            {label: 'GitHub', href: 'https://github.com/mu373/epydemix-webapi'},
            {label: 'epydemix', href: 'https://github.com/epistorm/epydemix'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Minami Ueda.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'json', 'python'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

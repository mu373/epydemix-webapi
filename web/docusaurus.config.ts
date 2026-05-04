import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'epydemix Web API',
  tagline: 'REST API for epidemic simulations',
  favicon: undefined,

  url: 'https://epydemix-webapi.vercel.app',
  baseUrl: '/',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

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
          editUrl: 'https://github.com/mu373/epydemix-webapi/blob/main/web/',
          sidebarCollapsible: false,
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          // Sort items inside the Release Notes category by semver descending
          // so per-version files don't need a hand-maintained `sidebar_position`.
          async sidebarItemsGenerator({defaultSidebarItemsGenerator, ...args}) {
            const items = await defaultSidebarItemsGenerator(args);
            const parseSemver = (slug: string): [number, number, number] => {
              const m = /^v(\d+)\.(\d+)\.(\d+)/.exec(slug);
              return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : [0, 0, 0];
            };
            const sortReleases = (nodes: any[]): any[] =>
              nodes.map((node) => {
                if (node.type === 'category' && node.label === 'Release Notes') {
                  const links = (node.items as any[]).filter(
                    (it) => it.type === 'doc' && /\/v\d/.test(it.id),
                  );
                  const others = (node.items as any[]).filter(
                    (it) => !(it.type === 'doc' && /\/v\d/.test(it.id)),
                  );
                  links.sort((a, b) => {
                    const av = parseSemver(a.id.split('/').pop() ?? '');
                    const bv = parseSemver(b.id.split('/').pop() ?? '');
                    for (let i = 0; i < 3; i++) {
                      if (av[i] !== bv[i]) return bv[i] - av[i];
                    }
                    return 0;
                  });
                  return {...node, items: [...others, ...links]};
                }
                if (node.type === 'category') {
                  return {...node, items: sortReleases(node.items as any[])};
                }
                return node;
              });
            return sortReleases(items);
          },
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
    './plugins/release-notes-data',
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

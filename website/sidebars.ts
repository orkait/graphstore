import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    'installation',
    {
      type: 'category',
      label: 'Concepts',
      collapsed: false,
      items: [
        'concepts/architecture',
        'concepts/remember-pipeline',
        'concepts/edge-matrix',
      ],
    },
    {
      type: 'category',
      label: 'DSL',
      collapsed: false,
      items: [
        'dsl/reference',
      ],
    },
    'query-builder',
    {
      type: 'category',
      label: 'Guides',
      collapsed: false,
      items: [
        'guides/first-memory',
        'guides/ingestion',
      ],
    },
    {
      type: 'category',
      label: 'Benchmarks',
      collapsed: false,
      items: [
        'benchmarks/overview',
      ],
    },
    'configuration',
  ],
};

export default sidebars;

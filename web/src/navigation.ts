interface NavItem {
  href: string;
  page: string;
  label: string;
  title: string;
}

interface NavGroup {
  label: string;
  items: NavItem[];
}

const groups: NavGroup[] = [
  {
    label: 'Create',
    items: [
      { href: '/', page: 'index.html', label: 'Creator', title: 'Compose and export model pairs' },
      { href: '/place.html', page: 'place.html', label: 'Place GLBs', title: 'Pose GLBs loaded from disk' },
    ],
  },
  {
    label: 'Inspect',
    items: [
      { href: '/viewer.html', page: 'viewer.html', label: 'Dataset', title: 'Inspect exported dataset samples' },
      { href: '/pipeline.html', page: 'pipeline.html', label: 'Pipeline', title: 'Inspect generated pipeline runs' },
      { href: '/internscenes.html', page: 'internscenes.html', label: 'InternScenes', title: 'Inspect downloaded InternScenes exports' },
    ],
  },
  {
    label: 'Evaluate',
    items: [
      { href: '/scene.html', page: 'scene.html', label: 'Scene Edits', title: 'Compare how LLMs edit a whole scene' },
      { href: '/placement.html', page: 'placement.html', label: 'Placement', title: 'Compare placement benchmark results' },
      { href: '/edit.html', page: 'edit.html', label: 'Edits', title: 'Inspect mesh edit results' },
      { href: '/segment.html', page: 'segment.html', label: 'PartField', title: 'Generate PartField mesh partitions' },
      { href: '/p3sam.html', page: 'p3sam.html', label: 'P3-SAM', title: 'Run P3-SAM automatic part segmentation' },
    ],
  },
];

const path = window.location.pathname.replace(/\/+$/, '');
const currentPage = path.split('/').pop() || 'index.html';
const links = groups
  .map(
    (group) => `
      <div class="app-nav__group">
        <span class="app-nav__group-label">${group.label}</span>
        ${group.items
          .map(
            (item) => `
              <a
                class="app-nav__link${item.page === currentPage ? ' active' : ''}"
                href="${item.href}"
                title="${item.title}"
                ${item.page === currentPage ? 'aria-current="page"' : ''}
              >${item.label}</a>
            `,
          )
          .join('')}
      </div>
    `,
  )
  .join('');

const header = document.createElement('header');
header.className = 'app-nav';
header.innerHTML = `
  <a class="app-nav__brand" href="/" aria-label="Data Creator home">
    <span class="app-nav__logo" aria-hidden="true">DC</span>
    <span class="app-nav__title">Data Creator</span>
  </a>
  <nav class="app-nav__links" aria-label="Tools">${links}</nav>
`;

document.body.prepend(header);

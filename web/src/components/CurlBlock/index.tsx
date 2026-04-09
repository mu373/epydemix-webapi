import BrowserOnly from '@docusaurus/BrowserOnly';
import CodeBlock from '@theme/CodeBlock';
import React, {useEffect, useState} from 'react';
import {EVENT_NAME, SERVERS, STORAGE_KEY} from '../EndpointSelector';

const PLACEHOLDER = 'BASE_URL';

function CurlBlockInner({children}: {children: string}) {
  const [endpoint, setEndpoint] = useState(SERVERS[0].url);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) setEndpoint(saved);

    const handler = (e: Event) => setEndpoint((e as CustomEvent<string>).detail);
    window.addEventListener(EVENT_NAME, handler);
    return () => window.removeEventListener(EVENT_NAME, handler);
  }, []);

  const code = children.trim().replace(new RegExp(PLACEHOLDER, 'g'), endpoint);
  return <CodeBlock language="bash">{code}</CodeBlock>;
}

export default function CurlBlock({children}: {children: string}) {
  const fallback = children.trim().replace(new RegExp(PLACEHOLDER, 'g'), SERVERS[0].url);
  return (
    <BrowserOnly fallback={<CodeBlock language="bash">{fallback}</CodeBlock>}>
      {() => <CurlBlockInner>{children}</CurlBlockInner>}
    </BrowserOnly>
  );
}

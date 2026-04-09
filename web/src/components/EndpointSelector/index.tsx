import BrowserOnly from '@docusaurus/BrowserOnly';
import React, {useEffect, useState} from 'react';
import styles from './styles.module.css';

export const SERVERS = [
  {url: 'https://epyscenario-api.isi.it/api/v1', description: 'Production'},
  {url: 'http://localhost:8000/api/v1', description: 'Local'},
];

export const STORAGE_KEY = 'epydemix-api-endpoint';
export const EVENT_NAME = 'epydemix-endpoint-change';

function EndpointSelectorInner() {
  const [endpoint, setEndpointState] = useState(SERVERS[0].url);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) setEndpointState(saved);

    const handler = (e: Event) => setEndpointState((e as CustomEvent<string>).detail);
    window.addEventListener(EVENT_NAME, handler);
    return () => window.removeEventListener(EVENT_NAME, handler);
  }, []);

  const setEndpoint = (url: string) => {
    setEndpointState(url);
    localStorage.setItem(STORAGE_KEY, url);
    window.dispatchEvent(new CustomEvent(EVENT_NAME, {detail: url}));
  };

  const isCustom = !SERVERS.some((s) => s.url === endpoint);

  return (
    <div className={styles.wrapper}>
      <span className={styles.label}>API endpoint</span>
      <div className={styles.buttons}>
        {SERVERS.map((server) => (
          <button
            key={server.url}
            className={`${styles.button} ${endpoint === server.url ? styles.active : ''}`}
            onClick={() => setEndpoint(server.url)}>
            {server.description}
          </button>
        ))}
        <input
          className={`${styles.input} ${isCustom ? styles.inputActive : ''}`}
          value={isCustom ? endpoint : ''}
          placeholder="Custom URL..."
          onChange={(e) => setEndpoint(e.target.value || SERVERS[0].url)}
        />
      </div>
    </div>
  );
}

export default function EndpointSelector() {
  return (
    <BrowserOnly fallback={<div />}>
      {() => <EndpointSelectorInner />}
    </BrowserOnly>
  );
}

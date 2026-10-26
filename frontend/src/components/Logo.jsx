import React, { useId } from 'react';

// Dhaaga's mark: a magnifying glass made from sewing. The lens is a loop of marigold
// thread, the handle is a needle, and a loose end trails out of the needle's eye.
// `animate` draws the thread once on load (skipped with prefers-reduced-motion).
export function LogoMark({ size = 46, animate = false, tail = true, title = 'Dhaaga' }) {
  const maskId = `needle-eye-${useId().replace(/:/g, '')}`;
  return (
    <svg
      className={`logo-mark ${animate ? 'logo-mark-animate' : ''}`}
      width={size}
      height={size}
      viewBox="0 0 64 64"
      {...(title ? { role: 'img', 'aria-label': title } : { 'aria-hidden': true })}
    >
      <defs>
        <mask id={maskId}>
          <rect width="64" height="64" fill="white" />
          <rect x="-3.2" y="-1" width="6.4" height="2" rx="1" fill="black" transform="translate(40.2 40.2) rotate(45)" />
        </mask>
      </defs>
      <path
        className="logo-needle"
        d="M33.24 38.76 A 3.9 3.9 0 0 1 38.76 33.24 L 60.5 60.5 Z"
        fill="#E2E8F0"
        mask={`url(#${maskId})`}
      />
      {tail && (
        <path
          className="logo-thread-tail"
          d="M40.2 40.2 C 46.5 38.4, 50 44.3, 54.2 43 C 57.6 41.9, 59.6 46.2, 62.4 45.4"
          pathLength="1"
          fill="none"
          stroke="#F5A524"
          strokeWidth="2.6"
          strokeLinecap="round"
        />
      )}
      <circle
        className="logo-thread-ring"
        cx="25"
        cy="25"
        r="14"
        pathLength="1"
        fill="none"
        stroke="#F5A524"
        strokeWidth="4.6"
        transform="rotate(45 25 25)"
      />
    </svg>
  );
}

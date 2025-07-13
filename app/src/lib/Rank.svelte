<script lang="ts">
  import type { Item } from "./core";

  let {
    items,
    left,
    right,
    onWinnerSelected,
  }: {
    items: Item[];
    left: number;
    right: number;
    onWinnerSelected: (winnerIndex: number) => void;
  } = $props();

  function selectIndex(selectedIndex: number) {
    onWinnerSelected(selectedIndex);
  }

  function handleKeyDown(e: KeyboardEvent) {
    // Prevent default behavior for our keys
    if (e.key.toLowerCase() === "f" || e.key.toLowerCase() === "j") {
      e.preventDefault();
    }

    if (e.key.toLowerCase() === "f") {
      selectIndex(left);
    } else if (e.key.toLowerCase() === "j") {
      selectIndex(right);
    }
  }
</script>

<svelte:window onkeydown={handleKeyDown} />

<main>
  <div class="items-wrapper">
    <button
      class="item item-1"
      onclick={() => selectIndex(left)}
      aria-label="Select left item"
    >
      <div class="item-content">
        <p>{items[left].text}</p>
      </div>
      <div class="item-number">F</div>
    </button>

    <div class="divider">
      <span>VS</span>
    </div>

    <button
      class="item item-2"
      onclick={() => selectIndex(right)}
      aria-label="Select right item"
    >
      <div class="item-content">
        <p>{items[right].text}</p>
      </div>
      <div class="item-number">J</div>
    </button>
  </div>

  <div class="controls">
    <p>Press <kbd>F</kbd> or <kbd>J</kbd> to select</p>
  </div>
</main>

<style lang="scss">
  main {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  .items-wrapper {
    flex: 1;
    display: flex;
    gap: 2rem;
    align-items: center;
    justify-content: center;

    @media (max-width: 768px) {
      flex-direction: column;
      gap: 1rem;
    }
  }

  .item {
    flex: 1;
    max-width: 400px;
    min-height: 300px;
    padding: 2rem;
    background-color: var(--col-bg);
    border: 2px solid color-mix(in srgb, var(--col-fg) 10%, transparent);
    border-radius: 0;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;

    &:hover {
      border-color: var(--col-accent);
      background-color: color-mix(in srgb, var(--col-accent) 5%, transparent);
      transform: scale(1.02);
    }

    &:active {
      transform: scale(0.98);
    }

    @media (max-width: 768px) {
      width: 100%;
      max-width: none;
      min-height: 200px;
    }
  }

  .item-content {
    text-align: center;

    p {
      margin: 0;
      font-size: 1.25rem;
      line-height: 1.6;
      color: var(--col-fg);
    }
  }

  .item-number {
    position: absolute;
    top: 1rem;
    right: 1rem;
    width: 2rem;
    height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: color-mix(in srgb, var(--col-fg) 10%, transparent);
    border-radius: 50%;
    font-weight: 600;
    font-size: 0.875rem;
    color: color-mix(in srgb, var(--col-fg) 70%, transparent);
  }

  .divider {
    display: flex;
    align-items: center;
    justify-content: center;

    span {
      font-weight: 600;
      color: color-mix(in srgb, var(--col-fg) 40%, transparent);
      font-size: 0.875rem;
      letter-spacing: 0.1em;
    }

    @media (max-width: 768px) {
      margin: 1rem 0;
    }
  }

  .controls {
    text-align: center;
    padding: 2rem 0;

    p {
      margin: 0;
      color: color-mix(in srgb, var(--col-fg) 60%, transparent);
      font-size: 0.875rem;
    }

    kbd {
      display: inline-block;
      padding: 0.2rem 0.5rem;
      margin: 0 0.2rem;
      background-color: color-mix(in srgb, var(--col-fg) 10%, transparent);
      border: 1px solid color-mix(in srgb, var(--col-fg) 20%, transparent);
      border-radius: 0.25rem;
      font-family: monospace;
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--col-fg);
    }
  }
</style>

<script lang="ts">
  import type { Item } from "./core";

  let items: Item[] = $state([]);
  let isDragging = $state(false);
  let removeHeader = $state(true);
  let fileInput: HTMLInputElement;

  let { onStartRanking }: { onStartRanking: (items: Item[]) => void } =
    $props();

  function handleFileSelect(file: File) {
    if (
      file &&
      (file.type === "text/plain" ||
        file.type === "text/csv" ||
        file.name.endsWith(".csv"))
    ) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target?.result as string;
        let lines = content
          .split("\n")
          .map((item) => item.trim())
          .filter((item) => item);

        // Remove header if it's a CSV file and the checkbox is checked
        const isCSV = file.type === "text/csv" || file.name.endsWith(".csv");
        if (isCSV && removeHeader && lines.length > 0) {
          lines = lines.slice(1);
        }

        // Replace items instead of adding
        items = lines.map((line) => ({ text: line }));
      };
      reader.readAsText(file);
    }
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    isDragging = false;

    const file = e.dataTransfer?.files[0];
    if (file) {
      handleFileSelect(file);
    }
  }

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    isDragging = true;
  }

  function handleDragLeave(e: DragEvent) {
    isDragging = false;
  }

  function handleClick() {
    fileInput.click();
  }

  function handleFileInputChange(e: Event) {
    const target = e.target as HTMLInputElement;
    const file = target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
    // Reset the input so the same file can be selected again
    target.value = "";
  }
</script>

<main>
  <h1>Fullrank</h1>

  <p>
    Lorem ipsum dolor sit amet consectetur adipisicing elit. Obcaecati
    cupiditate qui iure cum ipsam neque non, eum magnam aspernatur doloribus
    reprehenderit dignissimos officiis. Temporibus enim iste natus nobis
    perspiciatis eius!
  </p>

  <!-- Hidden file input -->
  <input
    type="file"
    accept=".txt,.csv,text/plain,text/csv"
    bind:this={fileInput}
    onchange={handleFileInputChange}
    style="display: none;"
  />

  <!-- File Drop Zone -->
  <div
    class="drop-zone"
    class:dragging={isDragging}
    ondrop={handleDrop}
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    onclick={handleClick}
    role="button"
    tabindex="0"
  >
    <svg
      width="48"
      height="48"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
      <polyline points="7 10 12 15 17 10"></polyline>
      <line x1="12" y1="15" x2="12" y2="3"></line>
    </svg>
    <p>Drop your list here or click to browse</p>
    <span>Line-separated text file or single-column CSV</span>

    <!-- CSV Option -->
    <label class="csv-option" onclick={(e) => e.stopPropagation()}>
      <input type="checkbox" bind:checked={removeHeader} />
      Remove header line from CSV files
    </label>
  </div>

  <!-- Ranking Button -->
  <div class="ranking-section">
    <button
      class="ranking-btn"
      disabled={items.length === 0}
      onclick={() => onStartRanking(items)}
    >
      Start Ranking ({items.length} items)
    </button>
  </div>
</main>

<style lang="scss">
  main {
    max-width: 65ch;
    margin: 0 auto;
    padding: 0 2rem;
  }

  h1 {
    margin-top: 6rem;
  }

  .drop-zone {
    margin: 2rem 0;
    border: 2px dashed color-mix(in srgb, var(--col-fg) 30%, transparent);
    border-radius: 0;
    padding: 3rem 2rem;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;

    &.dragging {
      border-color: var(--col-accent);
      background-color: color-mix(in srgb, var(--col-accent) 5%, transparent);
    }

    svg {
      margin: 0 auto 1rem;
      color: color-mix(in srgb, var(--col-fg) 50%, transparent);
    }

    p {
      margin: 0.5rem 0;
      font-weight: 500;
      color: var(--col-fg);
    }

    span {
      font-size: 0.875rem;
      color: color-mix(in srgb, var(--col-fg) 60%, transparent);
    }

    .csv-option {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      margin-top: 0.5rem;
      font-size: 0.875rem;
      color: color-mix(in srgb, var(--col-fg) 60%, transparent);
      cursor: pointer;

      input[type="checkbox"] {
        cursor: pointer;
      }
    }
  }

  .ranking-section {
    margin-top: 2rem;
  }

  .ranking-btn {
    width: 100%;
    padding: 0.75rem 2rem;
    background-color: var(--col-accent);
    color: var(--col-bg);
    border: none;
    border-radius: 0;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition:
      color 0.2s ease,
      background-color 0.2s ease;

    &:hover:not(:disabled) {
      background-color: color-mix(in srgb, var(--col-accent) 85%, black);
    }

    &:disabled {
      background-color: color-mix(in srgb, var(--col-fg) 10%, transparent);
      color: color-mix(in srgb, var(--col-fg) 40%, transparent);
      cursor: not-allowed;
    }
  }
</style>

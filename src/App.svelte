<script lang="ts">
  import type { Item } from "./lib/core";
  import Landing from "./lib/Landing.svelte";

  let items = $state<Item[] | null>(null);
  let model = $state<Model | null>(null);
</script>

<main>
  {#if items === null}
    <Landing
      onStartRanking={(items) => {
        items = items;
      }}
    />
  {:else}
    <h1>Fullrank</h1>
  {/if}
</main>

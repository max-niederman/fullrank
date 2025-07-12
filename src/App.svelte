<script lang="ts">
  import type { Item } from "./lib/core";
  import Landing from "./lib/Landing.svelte";
  import Rank from "./lib/Rank.svelte";

  let items = $state<Item[] | null>(null);
</script>

{#if items === null}
  <Landing
    onStartRanking={(newItems) => {
      items = newItems;
    }}
  />
{:else}
  <Rank
    {items}
    left={0}
    right={1}
    onWinnerSelected={(winnerIndex) => {
      console.log(winnerIndex);
    }}
  />
{/if}

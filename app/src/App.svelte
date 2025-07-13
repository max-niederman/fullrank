<script lang="ts">
  import { eye, zeros } from "vectorious";
  import { modelComparisons, type Comparison, type Item } from "./lib/core";
  import Landing from "./lib/Landing.svelte";
  import Rank from "./lib/Rank.svelte";

  let items = $state<Item[] | null>(null);
  let comparisons = $state<Comparison[]>([]);
  let index = $state(0);
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
    left={index}
    right={index + 1}
    onWinnerSelected={(winnerIndex) => {
      comparisons.push({
        winner: winnerIndex,
        loser: winnerIndex === index ? index + 1 : index,
      });
      index += 2;
      index %= items!.length;
      console.log(
        modelComparisons(
          zeros(items!.length),
          eye(items!.length),
          1,
          comparisons
        )
      );
    }}
  />
{/if}

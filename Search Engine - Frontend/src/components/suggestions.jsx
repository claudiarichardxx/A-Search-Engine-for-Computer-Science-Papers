import React, { useState } from 'react';
import suggestions from '../data.json'
const SearchSuggestions = ({clusterSearchInput,onInputSearchChange}) => {
 

  function onClickSuggestion(item){
      onInputSearchChange(item)
      
  }

  const filteredItems = suggestions.filter((element)=>element.suggestion.toLowerCase().startsWith(clusterSearchInput.toLowerCase())).slice(0,10)
  return (
    <ul className={ clusterSearchInput==''?'hidden':"absolute z-10 w-64 mt-2 bg-white border border-gray-300 rounded "}  >
      {filteredItems.map((item) => (
        <li
          key={item.index}
          className="px-4 py-2 cursor-pointer hover:bg-gray-100 text-black"
        onClick={()=>{onClickSuggestion(item)}}
        >
          {item.suggestion}
        </li>
      ))}
    </ul>
  );
};

export default SearchSuggestions;
